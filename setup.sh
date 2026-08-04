#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  RepoAgent One-Click Setup"
echo "=========================================="

# 1. 激活 venv
source /opt/venv/bin/activate
echo "[1/5] venv activated: $(which python)"

# 2. 装依赖（只在缺的时候装）
echo "[2/5] Checking dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -3

# 3. 卸载 flash-attn（ROCm 上用不了，每次容器重启都会回来）
echo "[3/5] Removing flash-attn (incompatible with ROCm)..."
pip uninstall -y flash-attn flash-attn-rotary 2>/dev/null || true

# 4. git pull 最新代码
echo "[4/5] Updating code..."
cd /workspace/RepoAMD
git pull -q || true

# 5. 设置环境变量
echo "[5/5] Setting ROCm env vars..."
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Terminal 1 - Start vLLM:"
echo "  cd /workspace/RepoAMD"
echo "  MODEL_PATH=./models/Qwen2.5-Coder-14B-Instruct bash start_llm.sh"
echo ""
echo "Terminal 2 - Start RepoAgent (after vLLM is ready):"
echo "  cd /workspace/RepoAMD"
echo "  python -m src.server /workspace/RepoAMD"
echo ""
