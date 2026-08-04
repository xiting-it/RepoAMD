#!/usr/bin/env bash
set -e
echo "=== RepoAgent Setup ==="

# Ensure venv
if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
    echo "[OK] venv activated: $(which python)"
fi

# Install all app deps
echo "=== Installing dependencies ==="
pip install -r requirements.txt 2>&1 | tail -3

# Uninstall flash-attn if present (breaks vLLM on ROCm)
pip uninstall flash-attn flash-attn-rotary -y 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Start vLLM:  MODEL_PATH=./models/Qwen2.5-Coder-14B-Instruct bash start_llm.sh"
echo "Start app:   python -m src.server /workspace/RepoAMD"
