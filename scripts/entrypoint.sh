#!/bin/bash
set -e

echo "=========================================="
echo "MinerU RunPod Serverless - Starting..."
echo "=========================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# 默认模型目录
MODELS_DIR="${MINERU_MODELS_DIR:-/workspace/models}"

# 检查是否有 RunPod 持久化卷
if [ -d "/runpod-volume" ]; then
    MODELS_DIR="/runpod-volume/mineru_models"
    export MINERU_MODELS_DIR="$MODELS_DIR"
    echo "[INFO] Using RunPod persistent volume: $MODELS_DIR"
fi

# 确保目录存在
mkdir -p "$MODELS_DIR"

# ============================================
# 【重要】先创建配置文件，再下载模型
# ============================================
export MINERU_MODEL_SOURCE=local

CONFIG_FILE="${HOME}/.mineru.json"
cat > "$CONFIG_FILE" << EOF
{
    "models-dir": {
        "pipeline": "${MODELS_DIR}",
        "vlm": "${MODELS_DIR}"
    },
    "config_version": "1.3.1"
}
EOF
echo "[INFO] Configuration file created at $CONFIG_FILE"

# 设置环境变量指向配置文件
export MINERU_TOOLS_CONFIG_JSON="$CONFIG_FILE"

# 检查模型是否存在
MODEL_CHECK_FILE="${MODELS_DIR}/.models_ready"

if [ ! -f "$MODEL_CHECK_FILE" ]; then
    echo "[INFO] Models not found, downloading..."
    echo "[INFO] This may take 10-30 minutes on first run..."

    # 下载模型
    /app/scripts/download_models.sh

    # 标记模型已就绪
    touch "$MODEL_CHECK_FILE"
    echo "[INFO] Models downloaded successfully"
else
    echo "[INFO] Models already present, skipping download"
fi

# 显示配置信息
echo "[INFO] Configuration:"
echo "  - Models directory: $MODELS_DIR"
echo "  - Model source: $MINERU_MODEL_SOURCE"
echo "  - Log level: ${MINERU_LOG_LEVEL:-INFO}"
echo "  - Max concurrent requests: ${MINERU_API_MAX_CONCURRENT_REQUESTS:-2}"
echo "  - Port: ${PORT:-8000}"

# 检查 GPU 可用性
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[WARN] nvidia-smi not found, GPU may not be available"
fi

echo "[INFO] Starting API server on port ${PORT:-8000}..."
echo "=========================================="

# 执行传入的命令
exec "$@"
