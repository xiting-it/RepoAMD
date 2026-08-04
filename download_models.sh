#!/usr/bin/env bash
# Download all models needed by RepoAgent.
#
# Usage:
#   bash download_models.sh                  # download all
#   bash download_models.sh --llm-only       # LLM only
#   bash download_models.sh --rag-only       # embedding + reranker only
#   MIRROR=modelscope bash download_models.sh   # use ModelScope (国内推荐)
#
# Models:
#   Qwen2.5-Coder-14B-Instruct   ~28 GB (FP16 weights)
#   BAAI/bge-m3                   ~2.4 GB (embedding, CPU)
#   BAAI/bge-reranker-v2-m3       ~0.6 GB (reranker, GPU)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MIRROR="${MIRROR:-hf-mirror}"   # hf-mirror (default) | modelscope | direct
mkdir -p "$MODEL_DIR"

# ── HF mirror ──
download_hf() {
    local repo=$1
    local dest=$2
    if command -v hf &>/dev/null; then
        hf download "$repo" --local-dir "$dest"
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$repo" --local-dir "$dest"
    else
        echo "Neither hf nor huggingface-cli found. Install: pip install huggingface_hub"
        exit 1
    fi
}

# ── ModelScope ──
download_ms() {
    local repo=$1
    local dest=$2
    python3 -c "
from modelscope import snapshot_download
snapshot_download('$repo', local_dir='$dest')
" 2>/dev/null || {
        echo "modelscope not installed or download failed. Install: pip install modelscope"
        echo "Falling back to HF mirror..."
        download_hf "$repo" "$dest"
    }
}

download() {
    local repo=$1
    local dest=$2
    case "$MIRROR" in
        modelscope)
            echo "  [source: ModelScope]"
            download_ms "$repo" "$dest"
            ;;
        direct)
            echo "  [source: HuggingFace direct]"
            download_hf "$repo" "$dest"
            ;;
        hf-mirror|*)
            echo "  [source: HF mirror (hf-mirror.com)]"
            export HF_ENDPOINT=https://hf-mirror.com
            download_hf "$repo" "$dest"
            ;;
    esac
}

download_llm() {
    echo "=== Downloading Qwen2.5-Coder-14B-Instruct (~28GB) ==="
    download Qwen/Qwen2.5-Coder-14B-Instruct "$MODEL_DIR/Qwen2.5-Coder-14B-Instruct"
    echo "LLM downloaded to $MODEL_DIR/Qwen2.5-Coder-14B-Instruct"
}

download_rag() {
    echo "=== Downloading BGE-m3 embedding model (~2.4GB) ==="
    download BAAI/bge-m3 "$MODEL_DIR/bge-m3"

    echo "=== Downloading bge-reranker-v2-m3 (~0.6GB) ==="
    download BAAI/bge-reranker-v2-m3 "$MODEL_DIR/bge-reranker-v2-m3"

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
echo "If download was slow, try: MIRROR=modelscope bash download_models.sh"
