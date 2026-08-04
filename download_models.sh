#!/usr/bin/env bash
# Download all models needed by RepoAgent.
#
# Usage:
#   bash download_models.sh                  # download all
#   bash download_models.sh --llm-only       # LLM only (for quick testing)
#   bash download_models.sh --rag-only       # embedding + reranker only
#
# Models:
#   Qwen2.5-Coder-14B-Instruct   ~28 GB (FP16 weights)
#   BAAI/bge-m3                   ~2.4 GB (embedding, CPU)
#   BAAI/bge-reranker-v2-m3       ~0.6 GB (reranker, GPU)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR"

download_llm() {
    echo "=== Downloading Qwen2.5-Coder-14B-Instruct (~28GB) ==="
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct \
            --local-dir "$MODEL_DIR/Qwen2.5-Coder-14B-Instruct"
    else
        echo "huggingface-cli not found. Install: pip install huggingface_hub"
        exit 1
    fi
    echo "LLM downloaded to $MODEL_DIR/Qwen2.5-Coder-14B-Instruct"
}

download_rag() {
    echo "=== Downloading BGE-m3 embedding model (~2.4GB) ==="
    huggingface-cli download BAAI/bge-m3 \
        --local-dir "$MODEL_DIR/bge-m3" || true

    echo "=== Downloading bge-reranker-v2-m3 (~0.6GB) ==="
    huggingface-cli download BAAI/bge-reranker-v2-m3 \
        --local-dir "$MODEL_DIR/bge-reranker-v2-m3" || true

    echo "RAG models downloaded to $MODEL_DIR/"
}

case "${1:-all}" in
    --llm-only) download_llm ;;
    --rag-only) download_rag ;;
    all|"")     download_llm; download_rag ;;
    *) echo "Unknown option: $1"; echo "Usage: bash download_models.sh [--llm-only|--rag-only]"; exit 1 ;;
esac

echo ""
echo "=== Done. Models in $MODEL_DIR/ ==="
echo "Update config.yaml model paths if you used a custom MODEL_DIR."
