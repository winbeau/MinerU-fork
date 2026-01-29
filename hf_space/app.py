"""
MinerU PDF 解析器 - HuggingFace Spaces ZeroGPU 版本
修复 H200 MIG (slice) CUBLAS 兼容性问题
"""

# ============================================
# 关键：在导入任何其他模块之前设置环境变量
# ============================================
import os
import sys

# 禁用多进程
os.environ['MINERU_WORKER_NUM'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 隐藏警告
os.environ['ONNXRUNTIME_LOG_SEVERITY_LEVEL'] = '3'

# 禁用 Flash Attention，强制 eager 模式
os.environ['ATTN_BACKEND'] = 'eager'
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'

# CUDA 设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================
# Monkey-patch ProcessPoolExecutor
# ============================================
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class FakeProcessPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        super().__init__(max_workers=max_workers, initializer=initializer, initargs=initargs)

concurrent.futures.ProcessPoolExecutor = FakeProcessPoolExecutor

import multiprocessing
import multiprocessing.pool

class FakePool:
    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None, context=None):
        self._executor = ThreadPoolExecutor(max_workers=processes)
    def map(self, func, iterable, chunksize=None):
        return list(self._executor.map(func, iterable))
    def starmap(self, func, iterable, chunksize=None):
        return list(self._executor.map(lambda args: func(*args), iterable))
    def apply(self, func, args=(), kwds={}):
        return self._executor.submit(func, *args, **kwds).result()
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

multiprocessing.Pool = FakePool
multiprocessing.pool.Pool = FakePool

print("✅ Monkey-patch: ProcessPoolExecutor → ThreadPoolExecutor")

# ============================================
# Patch Tensor.__matmul__ (@ 运算符) 使用 CPU fallback
# ============================================
import torch

# 禁用所有 SDPA 优化，强制使用 math 实现
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, 'enable_math_sdp'):
    torch.backends.cuda.enable_math_sdp(True)

print("✅ Disabled Flash/MemEfficient SDPA, using math SDPA only")

# 保存原始方法
_original_tensor_matmul = torch.Tensor.__matmul__
_original_matmul = torch.matmul
_original_bmm = torch.bmm
_cublas_error_count = 0

def _safe_matmul_impl(a, b, original_fn):
    """通用的安全矩阵乘法实现"""
    global _cublas_error_count
    try:
        return original_fn(a, b)
    except RuntimeError as e:
        if 'CUBLAS' in str(e):
            _cublas_error_count += 1
            if _cublas_error_count <= 5:
                print(f"⚠️ CUBLAS error #{_cublas_error_count}, falling back to CPU")
            # 回退到 CPU
            device = a.device
            dtype = a.dtype
            result = original_fn(a.float().cpu(), b.float().cpu())
            return result.to(device=device, dtype=dtype)
        raise

def safe_tensor_matmul(self, other):
    """安全的 @ 运算符"""
    return _safe_matmul_impl(self, other, _original_tensor_matmul)

def safe_matmul(input, other, *, out=None):
    """安全的 torch.matmul"""
    if out is not None:
        # 有 out 参数时不能简单回退
        return _original_matmul(input, other, out=out)
    return _safe_matmul_impl(input, other, _original_matmul)

def safe_bmm(input, mat2, *, out=None):
    """安全的 torch.bmm"""
    if out is not None:
        return _original_bmm(input, mat2, out=out)
    return _safe_matmul_impl(input, mat2, _original_bmm)

# 应用 patches
torch.Tensor.__matmul__ = safe_tensor_matmul
torch.matmul = safe_matmul
torch.bmm = safe_bmm

print("✅ Monkey-patch: Tensor.__matmul__/matmul/bmm with CPU fallback")

# ============================================
# 导入其他模块
# ============================================
import spaces
import gradio as gr
import tempfile
import time
from pathlib import Path


@spaces.GPU(duration=300)
def parse_document(
    file,
    backend: str = "vlm-auto-engine",
    lang: str = "ch",
    max_pages: int = 5,
    table_enable: bool = True,
    formula_enable: bool = True,
):
    """GPU 加速的文档解析函数"""
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # 再次确保 SDPA 设置正确
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
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
                for root, dirs, files in os.walk(output_dir):
                    for f in files:
                        if f.endswith('.md'):
                            with open(os.path.join(root, f), "r", encoding="utf-8") as file:
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
with gr.Blocks(title="MinerU PDF 解析器 (ZeroGPU)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 MinerU PDF 解析器
    ### 🚀 Powered by HuggingFace ZeroGPU (H200 Slice)

    将 PDF/图片转换为 Markdown，支持表格、公式识别。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="上传文件",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"],
            )

            backend = gr.Dropdown(
                choices=[
                    ("VLM 模式 (推荐)", "vlm-auto-engine"),
                    ("混合模式", "hybrid-auto-engine"),
                    ("Pipeline 模式", "pipeline"),
                ],
                value="vlm-auto-engine",
                label="解析后端",
            )

            lang = gr.Dropdown(
                choices=[
                    ("中文", "ch"),
                    ("英文", "en"),
                    ("自动检测", "auto"),
                ],
                value="ch",
                label="文档语言",
            )

            max_pages = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="最大页数")

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
    ### ⚠️ 说明
    - H200 MIG 分区可能存在 CUBLAS 兼容性问题
    - 如果解析失败，会自动回退到 CPU 计算（较慢但稳定）
    - 建议先用 1-3 页测试
    """)

if __name__ == "__main__":
    demo.launch()
