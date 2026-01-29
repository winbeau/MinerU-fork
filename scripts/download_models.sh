#!/bin/bash
set -e

echo "[INFO] ============================================"
echo "[INFO] MinerU Model Download Script"
echo "[INFO] ============================================"
echo "[INFO] Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# 模型源选择
MODEL_SOURCE="${MINERU_MODEL_SOURCE_DOWNLOAD:-huggingface}"

echo "[INFO] Model source: $MODEL_SOURCE"
echo "[INFO] Target directory: ${MINERU_MODELS_DIR:-/workspace/models}"

# 下载所有模型
echo "[INFO] Starting model download (this may take 10-30 minutes)..."

if [ "$MODEL_SOURCE" = "modelscope" ]; then
    echo "[INFO] Using ModelScope (recommended for China region)"
    mineru-models-download -s modelscope -m all
else
    echo "[INFO] Using HuggingFace"
    mineru-models-download -s huggingface -m all
fi

echo "[INFO] ============================================"
echo "[INFO] Model download completed!"
echo "[INFO] ============================================"

# 列出下载的模型
if [ -d "${MINERU_MODELS_DIR:-/workspace/models}" ]; then
    echo "[INFO] Downloaded models:"
    ls -la "${MINERU_MODELS_DIR:-/workspace/models}"
fi
