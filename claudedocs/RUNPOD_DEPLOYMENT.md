# RunPod Serverless 部署方案 - MinerU PDF 解析服务

> 生成时间：2025-01-29
> MinerU 版本：2.7.3
> 作者：Claude Code

## 📋 概述

基于 MinerU v2.7.3 仓库分析，设计了一套完整的 RunPod Serverless 部署方案，包含：
- 优化的 Dockerfile（GPU 加速 + 冷启动优化）
- 扩展的 FastAPI 服务（健康检查、异步任务、URL 解析）
- Dify 文档解析插件对接指南

---

## 1️⃣ 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     RunPod Serverless                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────────────┐    │
│  │   Dify      │───▶│  MinerU FastAPI Server              │    │
│  │   Plugin    │    │  ├─ /health                         │    │
│  └─────────────┘    │  ├─ /version                        │    │
│                     │  ├─ /parse (multipart/form-data)    │    │
│  ┌─────────────┐    │  ├─ /parse_url (JSON)               │    │
│  │   cURL /    │───▶│  ├─ /parse_async                    │    │
│  │   HTTP      │    │  └─ /tasks/{task_id}                │    │
│  └─────────────┘    └─────────────────────────────────────┘    │
│                                    │                            │
│                     ┌──────────────┴──────────────┐            │
│                     ▼                             ▼            │
│            ┌───────────────┐            ┌───────────────┐      │
│            │   Pipeline    │            │  Hybrid/VLM   │      │
│            │   Backend     │            │   Backend     │      │
│            │   (CPU OK)    │            │  (GPU Req.)   │      │
│            └───────────────┘            └───────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Network Volume (Models ~20GB)               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2️⃣ 文件结构

```
MinerU-fork/
├── Dockerfile.runpod              # RunPod 专用 Dockerfile
├── scripts/
│   ├── entrypoint.sh             # 容器启动脚本
│   └── download_models.sh        # 模型下载脚本
├── runpod_api/
│   ├── __init__.py
│   └── server.py                 # 扩展 FastAPI 服务
└── claudedocs/
    └── RUNPOD_DEPLOYMENT.md      # 本文档
```

---

## 3️⃣ API 设计详解

### 端点概览

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|------|
| `/health` | GET | 健康检查 | 否 |
| `/version` | GET | 版本和配置信息 | 否 |
| `/parse` | POST | 同步解析文件 | 可选 |
| `/parse_url` | POST | 从 URL 解析文件 | 可选 |
| `/parse_async` | POST | 异步解析（返回 task_id） | 可选 |
| `/tasks/{task_id}` | GET | 查询异步任务状态 | 可选 |
| `/docs` | GET | Swagger UI 文档 | 否 |

### 请求/响应示例

#### `POST /parse` - 同步解析

**请求：**
```bash
curl -X POST "https://your-runpod-endpoint.runpod.io/parse" \
  -H "Authorization: Bearer your-token" \
  -F "file=@document.pdf" \
  -F "backend=hybrid-auto-engine" \
  -F "max_pages=50" \
  -F "table_enable=true" \
  -F "formula_enable=true" \
  -F "lang=ch"
```

**响应：**
```json
{
  "status": "success",
  "markdown": "# 文档标题\n\n这是解析后的内容...\n\n| 表头1 | 表头2 |\n|-------|-------|\n| 数据1 | 数据2 |\n\n$$E=mc^2$$",
  "elapsed_ms": 15234,
  "page_count": 12,
  "backend": "hybrid-auto-engine",
  "version": "2.7.3",
  "content_list": [
    {"type": "title", "text": "文档标题", "page": 1},
    {"type": "paragraph", "text": "这是解析后的内容...", "page": 1},
    {"type": "table", "html": "<table>...</table>", "page": 2}
  ],
  "images": null
}
```

#### `POST /parse_url` - 从 URL 解析

**请求：**
```bash
curl -X POST "https://your-runpod-endpoint.runpod.io/parse_url" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "backend": "hybrid-auto-engine",
    "max_pages": 100,
    "table_enable": true,
    "formula_enable": true,
    "lang": "en"
  }'
```

**响应：** 同 `/parse`

#### `POST /parse_async` - 异步解析

**请求：**
```bash
curl -X POST "https://your-runpod-endpoint.runpod.io/parse_async" \
  -H "Authorization: Bearer your-token" \
  -F "file=@large_document.pdf" \
  -F "backend=hybrid-auto-engine"
```

**响应：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Task created successfully"
}
```

#### `GET /tasks/{task_id}` - 查询任务状态

**请求：**
```bash
curl "https://your-runpod-endpoint.runpod.io/tasks/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer your-token"
```

**响应（处理中）：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.6,
  "result": null,
  "error": null,
  "created_at": "2025-01-29T10:30:00.000Z",
  "updated_at": "2025-01-29T10:31:30.000Z"
}
```

**响应（完成）：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "status": "success",
    "markdown": "# 文档内容...",
    "elapsed_ms": 45000,
    "page_count": 150,
    "backend": "hybrid-auto-engine",
    "version": "2.7.3"
  },
  "error": null,
  "created_at": "2025-01-29T10:30:00.000Z",
  "updated_at": "2025-01-29T10:35:00.000Z"
}
```

#### `GET /health` - 健康检查

**响应：**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "models_loaded": true,
  "version": "2.7.3",
  "uptime_seconds": 3600.5
}
```

#### `GET /version` - 版本信息

**响应：**
```json
{
  "mineru_version": "2.7.3",
  "api_version": "1.0.0",
  "python_version": "3.10.12",
  "cuda_version": "12.1",
  "backends_available": [
    "pipeline",
    "vlm-auto-engine",
    "hybrid-auto-engine",
    "vlm-http-client",
    "hybrid-http-client"
  ],
  "config": {
    "max_concurrent_requests": 2,
    "max_file_size_mb": 100,
    "task_expire_hours": 24,
    "models_dir": "/workspace/models"
  }
}
```

---

## 4️⃣ Dify 对接说明

### 方式一：自定义工具（推荐）

在 Dify 中创建自定义工具，配置 API 调用：

#### 工具 Schema（OpenAPI 格式）

```yaml
openapi: 3.0.0
info:
  title: MinerU PDF Parser
  version: 1.0.0
servers:
  - url: https://your-runpod-endpoint.runpod.io

paths:
  /parse_url:
    post:
      operationId: parseDocumentFromUrl
      summary: 解析 PDF/图片文档
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - url
              properties:
                url:
                  type: string
                  description: 文档 URL
                backend:
                  type: string
                  enum: [pipeline, vlm-auto-engine, hybrid-auto-engine]
                  default: hybrid-auto-engine
                max_pages:
                  type: integer
                  minimum: 1
                  maximum: 1000
                table_enable:
                  type: boolean
                  default: true
                formula_enable:
                  type: boolean
                  default: true
                lang:
                  type: string
                  default: ch
      responses:
        '200':
          description: 解析成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  markdown:
                    type: string
                  page_count:
                    type: integer
                  elapsed_ms:
                    type: integer

security:
  - bearerAuth: []

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
```

#### Dify 配置步骤

1. **创建自定义工具**
   - 进入 Dify 控制台 → 工具 → 创建工具
   - 名称：`MinerU PDF Parser`
   - 粘贴上述 OpenAPI Schema

2. **配置认证**
   - 认证方式：Bearer Token
   - Token：您在环境变量 `MINERU_API_TOKEN` 中设置的值

3. **在工作流中使用**
   - 添加"工具"节点
   - 选择 `MinerU PDF Parser` → `parseDocumentFromUrl`
   - 输入参数：文档 URL

### 方式二：HTTP 请求节点

直接使用 Dify 的 HTTP 请求节点：

```yaml
# Dify HTTP 请求配置
Method: POST
URL: https://your-runpod-endpoint.runpod.io/parse_url
Headers:
  Authorization: Bearer {{api_token}}
  Content-Type: application/json
Body:
  {
    "url": "{{document_url}}",
    "backend": "hybrid-auto-engine",
    "max_pages": 100,
    "lang": "ch"
  }
```

### 最小 cURL 示例

```bash
# 1. 从 URL 解析
curl -X POST "https://your-runpod-endpoint.runpod.io/parse_url" \
  -H "Authorization: Bearer your-api-token" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/doc.pdf"}'

# 2. 上传文件解析
curl -X POST "https://your-runpod-endpoint.runpod.io/parse" \
  -H "Authorization: Bearer your-api-token" \
  -F "file=@document.pdf"

# 3. 健康检查（无需认证）
curl "https://your-runpod-endpoint.runpod.io/health"
```

---

## 5️⃣ 运行说明

### 本地开发

```bash
# 1. 克隆仓库
git clone https://github.com/your-fork/MinerU.git
cd MinerU

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -e ".[core,vllm]"
pip install aiofiles aiohttp

# 4. 下载模型（首次运行）
mineru-models-download -s huggingface -m all

# 5. 设置环境变量
export MINERU_MODEL_SOURCE=local
export MINERU_LOG_LEVEL=DEBUG
export MINERU_API_TOKEN=your-dev-token

# 6. 启动服务
python -m runpod_api.server --host 0.0.0.0 --port 8000 --reload
```

### Docker 构建

```bash
# 构建镜像（不预下载模型，使用持久化卷）
docker build -t mineru-runpod:latest -f Dockerfile.runpod .

# 构建镜像（预下载模型，镜像较大）
docker build -t mineru-runpod:full --build-arg PREDOWNLOAD_MODELS=1 -f Dockerfile.runpod .
```

### 本地 Docker 运行

```bash
# 运行（GPU）
docker run --gpus all -p 8000:8000 \
  -e MINERU_API_TOKEN=your-token \
  -e MINERU_LOG_LEVEL=INFO \
  -v /path/to/models:/workspace/models \
  mineru-runpod:latest

# 测试
curl http://localhost:8000/health
```

### RunPod 部署

1. **创建 Serverless Endpoint**
   - 登录 RunPod Console
   - Serverless → New Endpoint
   - Container Image: `your-registry/mineru-runpod:latest`
   - GPU Type: RTX 4090 / A100 / L40S（推荐）
   - Min Workers: 0（按需启动）
   - Max Workers: 根据负载设置

2. **配置 Network Volume（推荐）**
   - 创建 Network Volume（至少 30GB）
   - 挂载路径：`/runpod-volume`
   - 模型会自动保存到此卷，避免重复下载

3. **环境变量**
   ```
   MINERU_API_TOKEN=your-secure-token
   MINERU_LOG_LEVEL=INFO
   MINERU_API_MAX_CONCURRENT_REQUESTS=2
   MINERU_MODELS_DIR=/runpod-volume/mineru_models
   ```

4. **Build Context 设置**
   - Repository: 您的 GitHub 仓库 URL
   - Build Context: `/`（仓库根目录）
   - Dockerfile Path: `Dockerfile.runpod`

### 关键环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `MINERU_API_TOKEN` | 空 | API 认证 Token（空则禁用认证） |
| `MINERU_LOG_LEVEL` | INFO | 日志级别：DEBUG/INFO/WARNING/ERROR |
| `MINERU_MODEL_SOURCE` | local | 模型源：local/huggingface/modelscope |
| `MINERU_MODELS_DIR` | /workspace/models | 模型存储目录 |
| `MINERU_API_MAX_CONCURRENT_REQUESTS` | 2 | 最大并发请求数 |
| `MINERU_MAX_FILE_SIZE_MB` | 100 | 最大文件大小（MB） |
| `MINERU_DEVICE_MODE` | gpu | 设备模式：gpu/cpu |
| `MINERU_API_ENABLE_FASTAPI_DOCS` | 1 | 启用 Swagger 文档 |

---

## 6️⃣ 常见错误及排查

### 错误 1: 模型未找到

```
Error: Model not found at /workspace/models/...
```

**解决：**
```bash
# 检查模型目录
ls -la $MINERU_MODELS_DIR

# 手动下载
mineru-models-download -s huggingface -m all

# 或检查持久化卷是否正确挂载
```

### 错误 2: GPU 内存不足

```
RuntimeError: CUDA out of memory
```

**解决：**
- 降低 `MINERU_API_MAX_CONCURRENT_REQUESTS` 为 1
- 使用更大显存的 GPU（推荐 >= 16GB）
- 使用 `pipeline` 后端（显存需求更低）

### 错误 3: 冷启动超时

```
Error: Worker timeout during cold start
```

**解决：**
- 使用 Network Volume 预加载模型
- 预构建包含模型的完整镜像
- 增加 RunPod 的超时设置

### 错误 4: 依赖冲突

```
ERROR: pip's dependency resolver produced a conflict
```

**解决：**
```bash
# 使用基础镜像的 pip 环境
pip install --no-deps mineru
# 然后单独安装缺失依赖
```

### 错误 5: 文件类型不支持

```
HTTPException: 400 - Unsupported file type: docx
```

**解决：**
- MinerU 仅支持 PDF 和图片（PNG/JPG/JPEG/WEBP/GIF/BMP/TIFF）
- 对于 DOCX 等格式，需要先转换为 PDF

### 错误 6: 认证失败

```
HTTPException: 401 - Invalid token
```

**解决：**
- 检查 `Authorization` 头格式：`Bearer <token>`
- 确认 token 与 `MINERU_API_TOKEN` 环境变量一致
- 如不需要认证，可将 `MINERU_API_TOKEN` 设为空

---

## 7️⃣ 性能优化建议

### 冷启动优化

| 策略 | 预期效果 | 适用场景 |
|------|----------|----------|
| Network Volume 存储模型 | 冷启动 30s → 10s | 生产环境 |
| 预构建完整镜像 | 冷启动 30s → 5s | 高频使用 |
| 保持最小 Worker >= 1 | 无冷启动 | 预算充足 |

### 推理优化

| 策略 | 说明 |
|------|------|
| 使用 `hybrid-auto-engine` | 最佳精度/速度平衡 |
| 限制 `max_pages` | 减少大文档处理时间 |
| 使用异步 API | 允许批量处理，提高吞吐 |

### 成本优化

| GPU 类型 | VRAM | 价格/小时 | 推荐场景 |
|----------|------|-----------|----------|
| RTX 4090 | 24GB | $0.44 | 日常使用 |
| A100 40GB | 40GB | $1.99 | 大文档/批量 |
| L40S | 48GB | $1.14 | 高并发 |

---

## 8️⃣ 后端选择指南

| 后端 | 精度 | 速度 | VRAM | 语言支持 | 推荐场景 |
|------|------|------|------|----------|----------|
| `pipeline` | ⭐⭐⭐ | ⭐⭐⭐⭐ | 6GB | 多语言 | CPU 环境、快速处理 |
| `hybrid-auto-engine` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 10GB | 多语言 | **推荐**、平衡精度速度 |
| `vlm-auto-engine` | ⭐⭐⭐⭐⭐ | ⭐⭐ | 8GB | 中英文 | 最高精度需求 |
| `*-http-client` | 同上 | 取决于服务器 | 低 | 同上 | 远程 GPU 服务 |

---

## 9️⃣ 安全建议

1. **始终设置 `MINERU_API_TOKEN`** - 防止未授权访问
2. **使用 HTTPS** - RunPod 默认提供 SSL
3. **限制文件大小** - 设置合理的 `MINERU_MAX_FILE_SIZE_MB`
4. **定期更新镜像** - 获取安全补丁
5. **监控日志** - 检测异常访问模式

---

## 🔟 参考链接

- [MinerU 官方文档](https://opendatalab.github.io/MinerU/)
- [RunPod Serverless 文档](https://docs.runpod.io/serverless)
- [Dify 自定义工具文档](https://docs.dify.ai/guides/tools)
- [vLLM 官方镜像](https://hub.docker.com/r/vllm/vllm-openai)
