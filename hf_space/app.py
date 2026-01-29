"""
MinerU PDF 解析器 - HuggingFace Spaces ZeroGPU 版本
使用 monkey-patch 解决 daemonic processes 问题
"""

# ============================================
# 关键：在导入任何其他模块之前进行 monkey-patch
# ============================================
import os
import sys

# 禁用多进程相关环境变量
os.environ['MINERU_WORKER_NUM'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Monkey-patch: 将 ProcessPoolExecutor 替换为 ThreadPoolExecutor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# 保存原始的 ProcessPoolExecutor
_OriginalProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor

# 创建一个假的 ProcessPoolExecutor，实际使用 ThreadPoolExecutor
class FakeProcessPoolExecutor(ThreadPoolExecutor):
    """用 ThreadPoolExecutor 替代 ProcessPoolExecutor，避免 daemon 进程问题"""
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        # 忽略 mp_context 参数，因为 ThreadPoolExecutor 不需要
        super().__init__(max_workers=max_workers, initializer=initializer, initargs=initargs)

# 替换
concurrent.futures.ProcessPoolExecutor = FakeProcessPoolExecutor

# 同时替换 multiprocessing.Pool
import multiprocessing
import multiprocessing.pool

class FakePool:
    """用线程模拟 multiprocessing.Pool"""
    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None, context=None):
        self._executor = ThreadPoolExecutor(max_workers=processes)

    def map(self, func, iterable, chunksize=None):
        return list(self._executor.map(func, iterable))

    def starmap(self, func, iterable, chunksize=None):
        def wrapper(args):
            return func(*args)
        return list(self._executor.map(wrapper, iterable))

    def apply(self, func, args=(), kwds={}):
        future = self._executor.submit(func, *args, **kwds)
        return future.result()

    def apply_async(self, func, args=(), kwds={}, callback=None, error_callback=None):
        future = self._executor.submit(func, *args, **kwds)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def close(self):
        self._executor.shutdown(wait=False)

    def terminate(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def join(self):
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        return False

# 替换 multiprocessing.Pool
multiprocessing.Pool = FakePool
multiprocessing.pool.Pool = FakePool

print("✅ Monkey-patch applied: ProcessPoolExecutor → ThreadPoolExecutor")

# ============================================
# 现在可以安全导入其他模块
# ============================================
import spaces
import gradio as gr
import tempfile
import time
from pathlib import Path


@spaces.GPU(duration=300)
def parse_document(
    file,
    backend: str = "pipeline",
    lang: str = "ch",
    max_pages: int = 20,
    table_enable: bool = True,
    formula_enable: bool = True,
):
    """GPU 加速的文档解析函数"""
    import torch

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
        from mineru.cli.common import do_parse, read_fn
        from mineru.version import __version__

        with tempfile.TemporaryDirectory() as output_dir:
            file_path = Path(file.name if hasattr(file, 'name') else file)
            pdf_bytes = read_fn(file_path)
            file_stem = file_path.stem
            end_page = max_pages - 1 if max_pages else 99999

            os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

            print(f"📄 开始解析: {file_stem}")
            print(f"   Backend: {backend}, Language: {lang}, Max pages: {max_pages}")

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
                print(status)
                return status, markdown, elapsed
            else:
                # 查找可能的输出文件
                for root, dirs, files in os.walk(output_dir):
                    for f in files:
                        if f.endswith('.md'):
                            md_file = os.path.join(root, f)
                            with open(md_file, "r", encoding="utf-8") as file:
                                markdown = file.read()
                            return f"✅ 解析成功！耗时 {elapsed:.1f} 秒", markdown, elapsed
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

            max_pages = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="最大页数")

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
