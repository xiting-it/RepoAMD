# RepositoryAnalysisAgent

Privacy-first local code repository intelligence agent for AMD Radeon GPU.

RepositoryAnalysisAgent runs entirely on your local hardware. It uses AST-aware code indexing,
semantic retrieval with reranking, and a ReAct agent loop to answer questions
about codebases, locate bugs, and explain logic — all without sending a single
line of code to the cloud.

**Target hardware**: AMD Radeon PRO W7900 (48GB GDDR6, RDNA3 / gfx1100)
**Compute stack**: ROCm 7.2.1 + PyTorch + vLLM (or llama.cpp fallback)
**LLM**: Qwen2.5-Coder-14B-Instruct

---

## Architecture

```
Browser (localhost) → FastAPI API → Agent Engine (ReAct)
                                    ├── Tool Registry (search, read, grep, symbols)
                                    ├── Context Builder (dual budget 16K/32K)
                                    └── LLM Backend (vLLM / llama.cpp)

Index Pipeline: tree-sitter AST → BGE-m3 embedding (CPU) → ChromaDB
Search Pipeline: embedding retrieval (top-20) → bge-reranker (GPU) → top-15

GPU Layout:
  Process 1: vLLM server     ~42GB (weights 28 + KV pool 12 + runtime 2)
  Process 2: Reranker (GPU)   ~1.5GB
  CPU:        BGE-m3 embedding (indexing only)
```

## Prerequisites

**Recommended: use the pre-built Docker image**

The default environment already includes everything GPU-related:

```
10.5.10.89:1808/xinwei/radeon-cloud/vllm-dev:rocm7.2.1_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.16.0
```

This image contains: ROCm 7.2.1, Python 3.10, PyTorch 2.9, vLLM 0.16.0.
You only need to install app-level dependencies (fastapi, chromadb, tree-sitter, etc.).

- Docker image: vllm-dev:rocm7.2.1_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.16.0
- ROCm 7.2.1+ (`rocminfo` should show gfx1100)
- Python 3.10+
- ~32GB disk for models (14B FP16 + embedding + reranker)

## Installation

### 1. Pull the Docker image and start a container

```bash
docker pull 10.5.10.89:1808/xinwei/radeon-cloud/vllm-dev:rocm7.2.1_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.16.0

# Run with GPU access + mount your model storage
docker run -it --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /path/to/models:/models \
  -v /path/to/repos:/work \
  vllm-dev:rocm7.2.1_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.16.0
```

The image already has PyTorch 2.9 + vLLM 0.16.0 + ROCm 7.2.1. Do NOT reinstall them.

### 2. Clone and install app dependencies

```bash
git clone https://github.com/xiting-it/RepoAMD.git
cd RepoAMD
pip install -r requirements.txt
```

This installs only the lightweight app-level packages (fastapi, chromadb,
tree-sitter, sentence-transformers, etc.). Should take under 2 minutes.

### 3. Verify environment

```bash
python scripts/verify_rocm.py
```

## Usage

### 1. Start the LLM inference server

```bash
# Default: FP16 weights, enforce-eager (stable on gfx1100), qwen2 tool parser
bash start_llm.sh

# After M0 verifies graph mode is stable:
ENFORCE_EAGER=false bash start_llm.sh

# After M1 verifies FP8 is available (unlikely on gfx1100):
QUANT=fp8 bash start_llm.sh
```

Wait for "Application startup complete" (cold start: ~2-4 minutes for 14B FP16).

### 2. Start the RepositoryAnalysisAgent application

```bash
# Analyze a repository:
python -m src.server /path/to/your/repo

# Or use default (current directory):
python -m src.server

# Custom port:
python -m src.server /path/to/repo --port 3000
```

### 3. Open the web UI

Navigate to `http://127.0.0.1:8080` in your browser.

### 4. Index the repository

From the web UI, or via API:

```bash
curl -X POST http://127.0.0.1:8080/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/repo"}'
```

Indexing ~50 Python files on CPU takes ~8 minutes (one-time operation).

### 5. Ask questions

```
"What does the main entry point do?"
"Where is authentication handled?"
"Find the bug in the payment processing code"
"Trace how a user request flows through the system"
```

## Configuration

All settings are in `config.yaml`. Key options:

| Setting | Default | Notes |
|---|---|---|
| `server.host` | `127.0.0.1` | Privacy: loopback only |
| `llm.max_model_len` | `16384` | Upgrade to 32768 if FP8 verified |
| `llm.gpu_memory_utilization` | `0.88` | 14B FP16 needs 0.88 (42GB) |
| `llm.enforce_eager` | `true` | gfx1100 HIP graph unstable |
| `embedding.device` | `cpu` | One-time indexing, avoids VRAM contention |
| `reranker.device` | `cuda` | Online query hot path, ~1.5GB VRAM |
| `security.run_tests_enabled` | `false` | pytest collection = arbitrary code execution |

## AMD W7900 Optimization

RepositoryAnalysisAgent is designed specifically for RDNA3 / gfx1100:

- **GDDR6 bandwidth ceiling**: 864 GB/s ÷ 28GB (14B FP16) ≈ 31 tok/s theoretical max.
  Realistic expectation: 15-25 tok/s.
- **No FP8 acceleration**: RDNA3 WMMA supports FP8 storage but not the Matrix Core
  FP8 compute path that vLLM's ROCm kernels expect (written for gfx942/MI300X).
  Benchmark narrative focuses on "FP16 + retrieval strategy".
- **enforce-eager by default**: gfx1100 HIP graph capture is unreliable. The flag
  costs 10-20% throughput but prevents crashes.
- **ROCm stability env vars**: `HSA_ENABLE_SDMA=0`, `AMD_SERIALIZE_KERNEL=3`,
  `AMD_SERIALIZE_COPY=3` address known gfx1100 race conditions.

## Benchmark

```bash
python scripts/benchmark.py --base-url http://127.0.0.1:8000/v1
```

Measures: single-request throughput, batch concurrency, context length impact,
TTFT at various prompt sizes.

## Evaluation

```bash
# Run 20-question test suite + LLM-as-judge scoring
python scripts/eval_agent.py --app-url http://127.0.0.1:8080
```

Target: >70% of questions scored ≥2/3 by LLM judge.

## Project Structure

```
RepositoryAnalysisAgent/
├── config.yaml                 # All configuration
├── start_llm.sh                # vLLM server launcher (ROCm env + tool flags)
├── download_models.sh          # Model downloader
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── server.py               # FastAPI entry point
│   ├── config.py               # Config loader (typed dataclasses)
│   ├── backend.py              # LLM backend abstraction (vLLM/llama.cpp)
│   ├── tool_parser.py          # Dual-format tool call parsing
│   ├── agent/
│   │   ├── engine.py           # ReAct loop (max 8 iterations)
│   │   ├── prompts.py          # System prompt templates
│   │   └── context.py          # Context builder (dual budget 16K/32K)
│   ├── index/
│   │   ├── parser.py           # tree-sitter AST chunk extraction
│   │   ├── embeddings.py       # BGE-m3 (CPU)
│   │   ├── reranker.py         # bge-reranker-v2-m3 (GPU)
│   │   ├── vector_store.py     # ChromaDB wrapper
│   │   └── indexer.py          # Full indexing pipeline
│   ├── tools/
│   │   ├── registry.py         # Tool registration + dispatch
│   │   ├── search.py           # search_code (semantic+rerank) + grep_code
│   │   ├── files.py            # read_file + list_directory
│   │   ├── ast_tools.py        # get_symbols + find_references
│   │   └── exec.py             # run_tests (disabled by default)
│   ├── api/
│   │   ├── routes.py           # FastAPI routes (SSE chat, index, health)
│   │   └── schemas.py          # Pydantic models
│   └── session/
│       └── manager.py          # JSON-based session persistence
├── scripts/
│   ├── verify_rocm.py          # M0: environment verification
│   ├── benchmark.py            # M5: AMD benchmark suite
│   └── eval_agent.py           # M3: 20-question evaluation
├── eval/
│   └── test_cases.json         # 20 test questions
├── static/
│   ├── index.html              # Web UI
│   ├── styles.css
│   └── app.js
├── tests/
│   ├── test_tool_parser.py     # Tool call parsing tests
│   ├── test_parser.py          # tree-sitter parsing tests
│   └── test_config.py          # Config loading tests
├── spec.md                     # Product & engineering spec
└── plan.md                     # Implementation plan
```

## Troubleshooting

**`torch.cuda.is_available()` returns False**
- Ensure you installed the ROCm build of PyTorch (not the CUDA build)
- Check ROCm driver: `rocminfo | grep gfx`

**vLLM fails to compile or start**
- Try building from source with `PYTORCH_ROCM_ARCH=gfx1100`
- If persistent, fall back to llama.cpp (loses structured tool_calls)
- See `spec.md` section 6.3 and 7.3 for details

**HIP graph crashes**
- Keep `--enforce-eager` (default in `start_llm.sh`)
- Set `ENFORCE_EAGER=true` explicitly if overriding

**Inference produces NaN or inconsistent results**
- Set `AMD_SERIALIZE_KERNEL=3` and `AMD_SERIALIZE_COPY=3`
- These are already in `start_llm.sh` and `config.yaml`

**Out of memory (OOM)**
- Do not lower `gpu_memory_utilization` below 0.88 (14B FP16 needs ~30GB)
- Reranker runs in a separate process (~1.5GB) and does not count toward vLLM's budget

## Privacy

- API binds to `127.0.0.1` by default — no network exposure
- All inference, indexing, and search happen locally
- No telemetry, no external requests
- Verifiable with `tcpdump`: zero outbound traffic during operation

## License

MIT
