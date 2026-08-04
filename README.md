# RepoAgent

Privacy-first local code repository intelligence agent for AMD Radeon GPU.

RepoAgent runs entirely on your local hardware. It uses AST-aware code indexing,
semantic retrieval with reranking, and a ReAct agent loop to answer questions
about codebases, locate bugs, and explain logic — all without sending a single
line of code to the cloud.

**Target hardware**: AMD Radeon PRO W7900 (48GB GDDR6, RDNA3 / gfx1100)
**Compute stack**: ROCm 6.2 + PyTorch + vLLM (or llama.cpp fallback)
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

- AMD GPU: W7900 or similar RDNA3 card (gfx1100)
- ROCm 6.2+ (`rocminfo` should show gfx1100)
- Python 3.11+
- ~32GB disk for models (14B FP16 + embedding + reranker)

## Installation

### 1. Clone and install dependencies

```bash
git clone <your-repo-url> RepoAgent
cd RepoAgent
pip install -r requirements.txt
```

### 2. Install PyTorch (ROCm build)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

### 3. Install vLLM (ROCm)

vLLM does not have a standard PyPI wheel for ROCm. Use the ROCm-specific index:

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

Or build from source (30min-2h):

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
pip install -e . --rocm-version=6.2
```

> **Pin the version**: After verifying a working vLLM build, pin it in `requirements.txt`.
> ROCm support changes frequently — an unversioned install may break on update.

### 4. Verify environment (M0)

```bash
python scripts/verify_rocm.py
```

This checks: ROCm driver, HIP toolchain, PyTorch+ROCm, MIOpen attention backend,
inference stability, and vLLM import. All must pass before proceeding.

### 5. Download models

```bash
bash download_models.sh
# Or separately:
bash download_models.sh --llm-only    # Qwen2.5-Coder-14B (~28GB)
bash download_models.sh --rag-only    # BGE-m3 + reranker (~3GB)
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

### 2. Start the RepoAgent application

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

RepoAgent is designed specifically for RDNA3 / gfx1100:

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
RepoAgent/
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
