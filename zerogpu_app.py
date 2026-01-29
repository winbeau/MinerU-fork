"""
MinerU PDF 解析器 - HuggingFace Spaces ZeroGPU 版本
使用 @spaces.GPU 装饰器确保 GPU 正确分配
"""

import spaces
import gradio as gr
import tempfile
import os
import time
from pathlib import Path


@spaces.GPU(duration=180)  # 3分钟，大文档可能需要更长时间
def parse_document(
    file,
    backend: str = "hybrid-auto-engine",
    lang: str = "ch",
    max_pages: int = 50,
    table_enable: bool = True,
    formula_enable: bool = True,
):
    """
    GPU 加速的文档解析函数
    必须用 @spaces.GPU 装饰才能使用 H200
    """
    import torch

    # 打印 GPU 信息确认
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("❌ No GPU available!")
        return "错误：GPU 不可用", "", 0

    # 在 GPU 函数内部导入 MinerU（避免在 CPU 上初始化模型）
    from mineru.cli.common import do_parse, read_fn
    from mineru.version import __version__

    if file is None:
        return "请上传 PDF 或图片文件", "", 0

    start_time = time.time()

    try:
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as output_dir:
            # 读取文件
            file_path = Path(file.name if hasattr(file, 'name') else file)
            pdf_bytes = read_fn(file_path)
            file_stem = file_path.stem

            # 设置页数限制
            end_page = max_pages - 1 if max_pages else 99999

            # 设置环境变量
            os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

            print(f"📄 开始解析: {file_stem}")
            print(f"   Backend: {backend}")
            print(f"   Language: {lang}")
            print(f"   Max pages: {max_pages}")

            # 调用 MinerU 解析
            do_parse(
                output_dir=output_dir,
                pdf_file_names=[file_stem],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[lang],
                backend=backend,
                parse_method="auto",
                formula_enable=formula_enable,
                table_enable=table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                start_page_id=0,
                end_page_id=end_page,
            )

            # 确定结果路径
            if backend == "pipeline":
                result_dir = os.path.join(output_dir, file_stem, "auto")
            elif backend.startswith("vlm"):
                result_dir = os.path.join(output_dir, file_stem, "vlm")
            else:  # hybrid
                result_dir = os.path.join(output_dir, file_stem, "hybrid_auto")

            md_path = os.path.join(result_dir, f"{file_stem}.md")

            elapsed = time.time() - start_time

            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown = f.read()

                status = f"✅ 解析成功！耗时 {elapsed:.1f} 秒 (MinerU v{__version__}, GPU: {gpu_name})"
                print(status)
                return status, markdown, elapsed
            else:
                return f"❌ 解析失败：未找到输出文件", "", elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"❌ 解析错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "", elapsed


# Gradio 界面
with gr.Blocks(title="MinerU PDF 解析器 (ZeroGPU H200)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 MinerU PDF 解析器
    ### 🚀 Powered by HuggingFace ZeroGPU (NVIDIA H200 70GB)

    将 PDF/图片转换为 Markdown 格式，支持表格、公式识别。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="上传文件",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"],
            )

            backend = gr.Dropdown(
                choices=[
                    ("混合模式 (推荐)", "hybrid-auto-engine"),
                    ("VLM 模式 (最高精度)", "vlm-auto-engine"),
                    ("Pipeline 模式 (快速)", "pipeline"),
                ],
                value="hybrid-auto-engine",
                label="解析后端",
            )

            lang = gr.Dropdown(
                choices=[
                    ("中文", "ch"),
                    ("英文", "en"),
                    ("自动检测", "auto"),
                    ("日文", "japan"),
                    ("韩文", "korean"),
                    ("拉丁语系", "latin"),
                ],
                value="ch",
                label="文档语言",
            )

            max_pages = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="最大页数",
            )

            with gr.Row():
                table_enable = gr.Checkbox(value=True, label="表格识别")
                formula_enable = gr.Checkbox(value=True, label="公式识别")

            btn = gr.Button("🚀 开始解析", variant="primary", size="lg")

        with gr.Column(scale=2):
            status = gr.Textbox(label="状态", interactive=False)
            elapsed = gr.Number(label="耗时 (秒)", interactive=False)
            output = gr.Markdown(label="解析结果")

    btn.click(
        fn=parse_document,
        inputs=[file_input, backend, lang, max_pages, table_enable, formula_enable],
        outputs=[status, output, elapsed],
    )

    gr.Markdown("""
    ---
    ### 📝 说明
    - **混合模式**: 综合精度和速度，推荐大多数场景
    - **VLM 模式**: 最高精度，适合复杂文档
    - **Pipeline 模式**: 最快速度，适合简单文档

    ### ⚠️ 注意
    - ZeroGPU 有使用配额限制
    - 大文档可能需要较长处理时间
    """)

if __name__ == "__main__":
    demo.launch()
