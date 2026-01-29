"""
MinerU PDF 解析器 - HuggingFace Spaces ZeroGPU 版本
修复 daemonic processes 错误
"""

# 必须在最开始设置，禁用多进程
import os
os.environ['MINERU_WORKER_NUM'] = '0'  # 禁用 worker 进程
os.environ['OMP_NUM_THREADS'] = '1'  # 限制 OpenMP 线程
os.environ['MKL_NUM_THREADS'] = '1'  # 限制 MKL 线程

import spaces
import gradio as gr
import tempfile
import time
from pathlib import Path


@spaces.GPU(duration=300)  # 5分钟，单进程可能更慢
def parse_document(
    file,
    backend: str = "pipeline",  # 默认用 pipeline，更稳定
    lang: str = "ch",
    max_pages: int = 20,
    table_enable: bool = True,
    formula_enable: bool = True,
):
    """
    GPU 加速的文档解析函数
    使用单进程模式避免 daemonic process 错误
    """
    import torch

    # 确认 GPU 可用
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("❌ No GPU available!")
        return "错误：GPU 不可用", "", 0

    if file is None:
        return "请上传 PDF 或图片文件", "", 0

    start_time = time.time()

    try:
        # 在 GPU 函数内部导入，避免提前初始化
        from mineru.cli.common import read_fn
        from mineru.version import __version__

        # 使用更底层的 API 避免多进程
        from mineru.pdf_parser import PDFParser

        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as output_dir:
            # 读取文件
            file_path = Path(file.name if hasattr(file, 'name') else file)
            pdf_bytes = read_fn(file_path)
            file_stem = file_path.stem

            print(f"📄 开始解析: {file_stem}")
            print(f"   Backend: {backend}")
            print(f"   Language: {lang}")
            print(f"   Max pages: {max_pages}")

            # 设置解析参数
            end_page = max_pages - 1 if max_pages else 99999

            # 使用 PDFParser 直接解析（单进程）
            parser = PDFParser(
                pdf_bytes=pdf_bytes,
                model_backend=backend,
                lang=lang,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )

            # 解析文档
            result = parser.parse(
                start_page=0,
                end_page=end_page,
            )

            elapsed = time.time() - start_time

            # 获取 Markdown 结果
            if hasattr(result, 'get_markdown'):
                markdown = result.get_markdown()
            elif hasattr(result, 'markdown'):
                markdown = result.markdown
            elif isinstance(result, str):
                markdown = result
            else:
                # 尝试从 result 提取内容
                markdown = str(result)

            status = f"✅ 解析成功！耗时 {elapsed:.1f} 秒 (MinerU v{__version__}, GPU: {gpu_name})"
            print(status)
            return status, markdown, elapsed

    except ImportError as e:
        # 如果 PDFParser 不可用，回退到 do_parse
        print(f"⚠️ PDFParser 不可用，尝试 do_parse: {e}")
        return parse_with_do_parse(file, backend, lang, max_pages, table_enable, formula_enable, start_time)

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"❌ 解析错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "", elapsed


def parse_with_do_parse(file, backend, lang, max_pages, table_enable, formula_enable, start_time):
    """回退方案：使用 do_parse"""
    import torch

    try:
        from mineru.cli.common import do_parse, read_fn
        from mineru.version import __version__

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"

        with tempfile.TemporaryDirectory() as output_dir:
            file_path = Path(file.name if hasattr(file, 'name') else file)
            pdf_bytes = read_fn(file_path)
            file_stem = file_path.stem
            end_page = max_pages - 1 if max_pages else 99999

            # 设置环境变量禁用并行
            os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

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
            else:
                result_dir = os.path.join(output_dir, file_stem, "hybrid_auto")

            md_path = os.path.join(result_dir, f"{file_stem}.md")
            elapsed = time.time() - start_time

            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown = f.read()
                status = f"✅ 解析成功！耗时 {elapsed:.1f} 秒 (MinerU v{__version__}, GPU: {gpu_name})"
                return status, markdown, elapsed
            else:
                return f"❌ 解析失败：未找到输出文件", "", elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        traceback.print_exc()
        return f"❌ 解析错误: {str(e)}", "", elapsed


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
                    ("Pipeline 模式 (推荐)", "pipeline"),
                    ("混合模式", "hybrid-auto-engine"),
                    ("VLM 模式 (高精度)", "vlm-auto-engine"),
                ],
                value="pipeline",
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
                maximum=50,
                value=10,
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
    - **Pipeline 模式**: 最稳定，推荐 ZeroGPU 使用
    - **混合模式**: 综合精度和速度
    - **VLM 模式**: 最高精度，适合复杂文档

    ### ⚠️ 注意
    - ZeroGPU 有使用配额限制
    - 建议先用小文档测试
    """)

if __name__ == "__main__":
    demo.launch()
