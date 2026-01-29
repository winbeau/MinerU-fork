"""
MinerU RunPod Serverless API Server

扩展原有 FastAPI 服务，增加健康检查、版本、URL解析、异步任务等端点。
针对 RunPod Serverless 优化，支持 Dify 文档解析插件调用。

Usage:
    python -m runpod_api.server --host 0.0.0.0 --port 8000
"""

import os
import sys
import uuid
import time
import asyncio
import tempfile
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

import aiohttp
import aiofiles
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    Header,
    Depends,
)
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# MinerU imports
from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.version import __version__ as MINERU_VERSION
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes

# =============================================================================
# 配置
# =============================================================================

LOG_LEVEL = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

API_TOKEN = os.getenv("MINERU_API_TOKEN", "")  # 可选的 Bearer Token 认证
MAX_CONCURRENT = int(os.getenv("MINERU_API_MAX_CONCURRENT_REQUESTS", "2"))
OUTPUT_DIR = os.getenv("MINERU_OUTPUT_DIR", "/tmp/mineru_output")
MAX_FILE_SIZE = int(os.getenv("MINERU_MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
TASK_EXPIRE_HOURS = int(os.getenv("MINERU_TASK_EXPIRE_HOURS", "24"))

# =============================================================================
# 数据模型
# =============================================================================


class BackendType(str, Enum):
    """解析后端类型"""

    PIPELINE = "pipeline"
    VLM = "vlm-auto-engine"
    HYBRID = "hybrid-auto-engine"
    VLM_HTTP = "vlm-http-client"
    HYBRID_HTTP = "hybrid-http-client"


class TaskStatus(str, Enum):
    """异步任务状态"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ParseRequest(BaseModel):
    """解析请求参数"""

    backend: BackendType = Field(
        default=BackendType.HYBRID, description="解析后端类型"
    )
    max_pages: Optional[int] = Field(
        default=None, ge=1, le=1000, description="最大解析页数"
    )
    table_enable: bool = Field(default=True, description="启用表格识别")
    formula_enable: bool = Field(default=True, description="启用公式识别")
    lang: str = Field(default="ch", description="OCR 语言")
    parse_method: str = Field(default="auto", description="解析方法: auto/txt/ocr")


class ParseUrlRequest(ParseRequest):
    """URL 解析请求"""

    url: str = Field(..., description="文件 URL")
    filename: Optional[str] = Field(default=None, description="文件名（可选）")


class ParseResponse(BaseModel):
    """解析响应"""

    status: str
    markdown: Optional[str] = None
    elapsed_ms: int
    page_count: int
    backend: str
    version: str
    content_list: Optional[List[Dict]] = None
    images: Optional[Dict[str, str]] = None


class AsyncParseResponse(BaseModel):
    """异步解析响应"""

    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""

    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[ParseResponse] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    gpu_available: bool
    models_loaded: bool
    version: str
    uptime_seconds: float


class VersionResponse(BaseModel):
    """版本信息响应"""

    mineru_version: str
    api_version: str
    python_version: str
    cuda_version: Optional[str]
    backends_available: List[str]
    config: Dict[str, Any]


# =============================================================================
# 任务存储（生产环境建议使用 Redis）
# =============================================================================


class TaskStore:
    """内存任务存储"""

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, task_id: str) -> Dict[str, Any]:
        """创建新任务"""
        async with self._lock:
            now = datetime.utcnow().isoformat()
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            return self._tasks[task_id]

    async def update(self, task_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新任务状态"""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)
                self._tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
                return self._tasks[task_id]
            return None

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务"""
        async with self._lock:
            return self._tasks.get(task_id)

    async def cleanup_expired(self) -> int:
        """清理过期任务"""
        async with self._lock:
            now = datetime.utcnow()
            expired = []
            for task_id, task in self._tasks.items():
                created = datetime.fromisoformat(task["created_at"])
                if now - created > timedelta(hours=TASK_EXPIRE_HOURS):
                    expired.append(task_id)
            for task_id in expired:
                del self._tasks[task_id]
            return len(expired)


task_store = TaskStore()

# =============================================================================
# 全局变量
# =============================================================================

_semaphore: Optional[asyncio.Semaphore] = None
_start_time: float = 0


# =============================================================================
# 认证依赖
# =============================================================================


async def verify_token(authorization: Optional[str] = Header(None)):
    """可选的 Bearer Token 认证"""
    if not API_TOKEN:
        return True

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    token = authorization[7:]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    return True


# =============================================================================
# 应用生命周期
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _semaphore, _start_time
    _start_time = time.time()
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    logger.info(f"MinerU API Server starting - Version {MINERU_VERSION}")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 启动任务清理协程
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    # 取消清理任务
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    logger.info("MinerU API Server shutting down")


async def periodic_cleanup():
    """定期清理过期任务"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时清理一次
            cleaned = await task_store.cleanup_expired()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired tasks")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# =============================================================================
# 创建 FastAPI 应用
# =============================================================================

enable_docs = os.getenv("MINERU_API_ENABLE_FASTAPI_DOCS", "1") == "1"

app = FastAPI(
    title="MinerU PDF Parser API",
    description="PDF/图片解析服务 - RunPod Serverless",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if enable_docs else None,
    redoc_url="/redoc" if enable_docs else None,
    openapi_url="/openapi.json" if enable_docs else None,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 工具函数
# =============================================================================


def get_gpu_info() -> tuple[bool, Optional[str]]:
    """获取 GPU 信息"""
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except ImportError:
        return False, None


def check_models_loaded() -> bool:
    """检查模型是否已加载"""
    models_dir = os.getenv("MINERU_MODELS_DIR", "/workspace/models")
    return os.path.exists(os.path.join(models_dir, ".models_ready"))


async def download_file(
    url: str, filename: Optional[str] = None
) -> tuple[bytes, str]:
    """从 URL 下载文件"""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download file: HTTP {response.status}",
                )

            content = await response.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {len(content) / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE / 1024 / 1024}MB",
                )

            if not filename:
                # 从 URL 或 Content-Disposition 推断文件名
                cd = response.headers.get("Content-Disposition", "")
                if "filename=" in cd:
                    filename = cd.split("filename=")[1].strip('"\'')
                else:
                    filename = Path(url.split("?")[0]).name or "document.pdf"

            return content, filename


async def process_file_bytes(
    file_bytes: bytes,
    filename: str,
    params: ParseRequest,
) -> ParseResponse:
    """处理文件字节"""
    start_time = time.time()

    # 验证文件类型
    suffix = guess_suffix_by_bytes(file_bytes, Path(filename))
    if suffix not in pdf_suffixes + image_suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # 如果是图片，转换为 PDF
    if suffix in image_suffixes:
        from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes

        file_bytes = images_bytes_to_pdf_bytes(file_bytes)

    # 创建输出目录
    task_id = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(output_path, exist_ok=True)

    try:
        # 确定后端类型
        backend = params.backend.value

        # 设置页数限制
        end_page_id = params.max_pages - 1 if params.max_pages else 99999

        # 调用 MinerU 解析
        await aio_do_parse(
            output_dir=output_path,
            pdf_file_names=[Path(filename).stem],
            pdf_bytes_list=[file_bytes],
            p_lang_list=[params.lang],
            backend=backend,
            parse_method=params.parse_method,
            formula_enable=params.formula_enable,
            table_enable=params.table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            start_page_id=0,
            end_page_id=end_page_id,
        )

        # 读取结果
        pdf_name = Path(filename).stem
        if backend == "pipeline":
            result_dir = os.path.join(output_path, pdf_name, params.parse_method)
        elif backend.startswith("vlm"):
            result_dir = os.path.join(output_path, pdf_name, "vlm")
        else:  # hybrid
            result_dir = os.path.join(output_path, pdf_name, f"hybrid_{params.parse_method}")

        # 读取 Markdown
        md_path = os.path.join(result_dir, f"{pdf_name}.md")
        markdown = None
        if os.path.exists(md_path):
            async with aiofiles.open(md_path, "r", encoding="utf-8") as f:
                markdown = await f.read()

        # 读取 content_list
        cl_path = os.path.join(result_dir, f"{pdf_name}_content_list.json")
        content_list = None
        if os.path.exists(cl_path):
            async with aiofiles.open(cl_path, "r", encoding="utf-8") as f:
                content_list = json.loads(await f.read())

        # 统计页数
        import pypdfium2 as pdfium

        pdf_doc = pdfium.PdfDocument(file_bytes)
        page_count = len(pdf_doc)
        pdf_doc.close()

        elapsed_ms = int((time.time() - start_time) * 1000)

        return ParseResponse(
            status="success",
            markdown=markdown,
            elapsed_ms=elapsed_ms,
            page_count=min(page_count, params.max_pages or page_count),
            backend=backend,
            version=MINERU_VERSION,
            content_list=content_list,
        )

    except Exception as e:
        logger.exception(f"Parse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(output_path, ignore_errors=True)
        except Exception:
            pass


# =============================================================================
# API 端点
# =============================================================================


@app.get("/", tags=["System"])
async def root():
    """根路径"""
    return {
        "service": "MinerU PDF Parser",
        "version": MINERU_VERSION,
        "docs": "/docs" if enable_docs else "disabled",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    健康检查端点

    返回服务状态、GPU 可用性、模型加载状态等信息。
    """
    gpu_available, _ = get_gpu_info()
    models_loaded = check_models_loaded()

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        gpu_available=gpu_available,
        models_loaded=models_loaded,
        version=MINERU_VERSION,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/version", response_model=VersionResponse, tags=["System"])
async def get_version():
    """
    获取版本和配置信息

    返回 MinerU 版本、API 版本、可用后端等详细信息。
    """
    gpu_available, cuda_version = get_gpu_info()

    backends = ["pipeline"]
    if gpu_available:
        backends.extend(["vlm-auto-engine", "hybrid-auto-engine"])
    backends.extend(["vlm-http-client", "hybrid-http-client"])

    return VersionResponse(
        mineru_version=MINERU_VERSION,
        api_version="1.0.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        cuda_version=cuda_version,
        backends_available=backends,
        config={
            "max_concurrent_requests": MAX_CONCURRENT,
            "max_file_size_mb": MAX_FILE_SIZE // 1024 // 1024,
            "task_expire_hours": TASK_EXPIRE_HOURS,
            "models_dir": os.getenv("MINERU_MODELS_DIR", "/workspace/models"),
        },
    )


@app.post(
    "/parse",
    response_model=ParseResponse,
    tags=["Parse"],
    dependencies=[Depends(verify_token)],
)
async def parse_file(
    file: UploadFile = File(..., description="PDF 或图片文件"),
    backend: BackendType = Form(default=BackendType.HYBRID, description="解析后端"),
    max_pages: Optional[int] = Form(
        default=None, ge=1, le=1000, description="最大页数"
    ),
    table_enable: bool = Form(default=True, description="启用表格识别"),
    formula_enable: bool = Form(default=True, description="启用公式识别"),
    lang: str = Form(default="ch", description="OCR 语言"),
):
    """
    同步解析上传的 PDF/图片文件

    - **backend**: 解析后端类型
      - `pipeline`: 传统流水线，支持多语言，无幻觉
      - `hybrid-auto-engine`: 混合模式（推荐），高精度，支持多语言
      - `vlm-auto-engine`: 纯 VLM 模式，最高精度，仅支持中英文
    - **max_pages**: 限制解析的最大页数
    - **table_enable**: 是否启用表格识别
    - **formula_enable**: 是否启用公式识别
    - **lang**: OCR 语言（ch/en/latin 等）
    """
    if _semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail=f"Server at max capacity ({MAX_CONCURRENT} concurrent requests)",
        )

    async with _semaphore:
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(content) / 1024 / 1024:.1f}MB",
            )

        params = ParseRequest(
            backend=backend,
            max_pages=max_pages,
            table_enable=table_enable,
            formula_enable=formula_enable,
            lang=lang,
        )

        return await process_file_bytes(content, file.filename, params)


@app.post(
    "/parse_url",
    response_model=ParseResponse,
    tags=["Parse"],
    dependencies=[Depends(verify_token)],
)
async def parse_url(request: ParseUrlRequest):
    """
    从 URL 下载并解析文件

    - **url**: 文件的 HTTP/HTTPS URL
    - **filename**: 可选的文件名（用于确定文件类型）
    - 其他参数同 /parse 端点
    """
    if _semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail=f"Server at max capacity ({MAX_CONCURRENT} concurrent requests)",
        )

    async with _semaphore:
        content, filename = await download_file(request.url, request.filename)
        return await process_file_bytes(content, filename, request)


@app.post(
    "/parse_async",
    response_model=AsyncParseResponse,
    tags=["Async"],
    dependencies=[Depends(verify_token)],
)
async def parse_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF 或图片文件"),
    backend: BackendType = Form(default=BackendType.HYBRID, description="解析后端"),
    max_pages: Optional[int] = Form(
        default=None, ge=1, le=1000, description="最大页数"
    ),
    table_enable: bool = Form(default=True, description="启用表格识别"),
    formula_enable: bool = Form(default=True, description="启用公式识别"),
    lang: str = Form(default="ch", description="OCR 语言"),
):
    """
    异步解析文件，立即返回任务 ID

    使用 GET /tasks/{task_id} 查询解析进度和结果
    """
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content) / 1024 / 1024:.1f}MB",
        )

    task_id = str(uuid.uuid4())
    await task_store.create(task_id)

    params = ParseRequest(
        backend=backend,
        max_pages=max_pages,
        table_enable=table_enable,
        formula_enable=formula_enable,
        lang=lang,
    )

    # 在后台执行解析
    background_tasks.add_task(
        background_parse,
        task_id,
        content,
        file.filename,
        params,
    )

    return AsyncParseResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task created successfully",
    )


async def background_parse(
    task_id: str,
    content: bytes,
    filename: str,
    params: ParseRequest,
):
    """后台解析任务"""
    await task_store.update(task_id, status=TaskStatus.PROCESSING, progress=0.1)

    try:
        async with _semaphore:
            await task_store.update(task_id, progress=0.3)
            result = await process_file_bytes(content, filename, params)
            await task_store.update(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                result=result.model_dump(),
            )
    except Exception as e:
        logger.exception(f"Background parse failed: {e}")
        await task_store.update(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e),
        )


@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Async"],
    dependencies=[Depends(verify_token)],
)
async def get_task_status(task_id: str):
    """
    查询异步任务状态

    - **task_id**: 由 /parse_async 返回的任务 ID
    """
    task = await task_store.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        result=ParseResponse(**task["result"]) if task["result"] else None,
        error=task["error"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
    )


# =============================================================================
# 主入口
# =============================================================================


def main():
    """命令行入口"""
    import click

    @click.command()
    @click.option("--host", default="0.0.0.0", help="Server host")
    @click.option("--port", default=8000, type=int, help="Server port")
    @click.option("--reload", is_flag=True, help="Enable auto-reload")
    def run(host, port, reload):
        """启动 MinerU API 服务器"""
        uvicorn.run(
            "runpod_api.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=1,  # GPU 模型需要单进程
        )

    run()


if __name__ == "__main__":
    main()
