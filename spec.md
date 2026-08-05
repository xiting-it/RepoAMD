# RepositoryAnalysisAgent — 产品与工程规格书

> 版本: 3.2 | 日期: 2026-08-04
> 目标硬件: AMD Radeon PRO W7900 (48GB GDDR6, RDNA3 / gfx1100)
> ROCm 目标版本: 7.2.1 (需在 M0 阶段锁定具体小版本号)
> 赛事: AMD Pervasive AI Developer Contest — Track 2

---

## 1. 产品概述

### 1.1 产品定义

RepositoryAnalysisAgent 是一款**隐私优先的本地代码库智能助手**。它在用户本地硬件上运行，
通过 AST 感知的代码索引和检索增强生成 (RAG) 技术，让大语言模型理解整个代码仓库，
从而回答代码问题、定位 Bug、解释逻辑——全程代码不离开本机。

**本项目不包含代码补全 (FIM) 功能。**

### 1.2 核心价值主张

| 价值点 | 说明 |
|---|---|
| 零数据外泄 | 推理、检索、索引全部在本地完成，API 默认绑定 127.0.0.1 |
| 仓库级理解 | 理解整个仓库的结构、依赖、调用关系 |
| AST 感知 | 基于 tree-sitter 解析代码语法树 |
| AMD 原生优化 | 针对 Radeon GPU 的 ROCm/HIP 计算栈优化 |
| Agent 智能体 | 多步推理 + 工具调用，自主探索代码库 |

### 1.3 目标用户

- 企业开发团队 / 安全领域开发者 / 开源维护者 / 拥有 AMD GPU 的个人开发者

### 1.4 竞品分析

| 竞品 | 定位 | 与 RepositoryAnalysisAgent 的区别 |
|---|---|---|
| GitHub Copilot / Cursor | 云端编辑器 Agent | 代码上云 |
| Continue.dev | 开源，可接本地模型 | 有 @codebase 索引，但 Agent 自主性和 AMD GPU 优化不如本方案 |
| Tabby | 本地代码补全服务 | 聚焦补全 (FIM) |
| Aider | 终端 AI pair programmer | 有 tree-sitter repo map，但无向量检索、无多步 Agent |
| **RepositoryAnalysisAgent** | **本地 + 仓库级 Agent** | **AST 检索 + reranking + ReAct + AMD GPU 优化** |

---

## 2. 应用场景

1. 代码库快速理解 (新人 onboarding)
2. Bug 定位 (跨文件追踪根因)
3. 代码审查 (git diff 风险分析)
4. 跨文件影响分析
5. 测试生成

---

## 3. 系统架构

### 3.1 总体架构

```
+------------------------------------------------------------------+
|                        用户 (浏览器 localhost)                     |
+----------------------------+-------------------------------------+
                             | HTTP (默认 127.0.0.1)
                             v
+------------------------------------------------------------------+
|                    API 层 (FastAPI)                               |
|  /chat (Agent)  /index  /health  /workspace                      |
|       |           |            |                                   |
|       v           v            v                                   |
|  +---------+ +---------+ +--------+                                |
|  | Agent   | | Index   | | System |                                |
|  | Engine  | | Engine  | | Monitor|                                |
|  +----+----+ +----+----+ +--------+                                |
|       |     +-----+------+                                         |
|       |     | Tool Registry|                                       |
|       |     +-----+------+                                         |
+-------+-----------+------------------------------------------------+
        |           |
        v           v
+------------------------------------------------------------------+
|                   推理层 (AMD W7900 GPU, 48GB)                    |
|                                                                   |
|  进程 1: vLLM server (独立进程)                                   |
|    gpu_memory_utilization: 0.88 → 预分配 ~42GB                    |
|    内含: 权重 28GB + KV cache 池 ~12GB + 运行时 2GB               |
|                                                                   |
|  进程 2: FastAPI 应用 (含 reranker)                               |
|    Reranker 模型 ~0.6GB + HIP context ~0.9GB = ~1.5GB             |
|                                                                   |
|  CPU 侧: BGE-m3 embedding (索引时运行)                            |
|  总 GPU 占用: ~43.7GB | 剩余: ~4.3GB                              |
+------------------------------------------------------------------+
```

### 3.2 检索路径设计: Agent 自主检索 (实验性)

> **风险声明**: 14B 模型做多步工具调用的可靠性未经验证。
> M3 质量评估决定是否保留此设计或回退到预检索混合模式。

```
用户提问 -> Context Builder (系统prompt+仓库树+历史) -> Agent ReAct 循环:
  Round 1: search_code("auth") -> [embedding top-20 -> GPU rerank -> top-15]
  Round 2: read_file("src/auth/login.py")
  Round 3: get_symbols("src/utils/crypto.py")
  Round 4: LLM 综合分析 -> 最终回答
```

### 3.3 检索管线 (含 reranking)

```
search_code 内部:
  1. embedding 向量检索 (CPU) -> top-20 候选
  2. bge-reranker-v2-m3 cross-encoder (GPU) -> top-15
  3. 返回给 LLM
```

Reranker 必须在 GPU 上: CPU 上 20 candidates cross-encoder 约 4-7s/次，
Agent 每次提问调用 2-3 次 search_code = 8-21s 纯 reranker 延迟，不可接受。
GPU 上同样操作 < 0.5s。

### 3.4 分层职责

| 层 | 关键组件 |
|---|---|
| Presentation | 静态前端 (HTML/CSS/JS + highlight.js) |
| API | FastAPI + Uvicorn |
| Agent | 自研 Agent Engine (ReAct) |
| Indexing | tree-sitter + ChromaDB |
| Retrieval | BGE-m3 (CPU) + bge-reranker (GPU) |
| Inference | vLLM (主) / llama.cpp (备) |

---

## 4. 核心组件规格

### 4.1 Agent Engine

**ReAct 循环**: 最多 8 轮。每轮: 组装上下文 → 调用 LLM → 解析 tool_call → 执行工具 → 继续或结束。

**系统 Prompt**:

```
你是一个代码库分析专家。使用以下工具来探索和理解代码仓库。
当前仓库: {repo_path}
仓库结构: {repo_tree}
工具使用规则:
- 先搜索再读取
- 优先用 get_symbols 获取结构
- 回答引用文件路径和行号
- 不超过 4 次工具调用
- 信息够了就直接回答
```

**上下文窗口预算** (两套):

```python
BUDGET_32K = {
    "system_prompt": 1500, "repo_structure": 1000,
    "conversation": 4000, "tool_results": 17000, "response_margin": 9268,
}
BUDGET_16K = {
    "system_prompt": 1200, "repo_structure": 800,
    "conversation": 2000, "tool_results": 9000, "response_margin": 3384,
}
```

**工具调用**: Qwen2.5 在 vLLM 中需要显式启用:

```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

> 不加这两个 flag，vLLM 接受 tools 参数但模型输出不含结构化 tool_calls。
> Agent Engine 保留文本格式解析作为备选 (llama.cpp 路线唯一选项，见 6.3)。

### 4.2 Index Engine

文件遍历 (过滤 .git/node_modules 等) → tree-sitter AST 解析 → chunk 提取
→ embedding (CPU) → ChromaDB → 符号表。

**find_references**: 启发式符号引用查找，非精确调用图。
**增量索引**: SHA256 哈希检测变更。

### 4.3 Tool Registry

```
search_code     语义搜索 (含 reranking)
grep_code       精确文本/正则搜索 (ripgrep)
read_file       读取文件
list_directory  列目录
get_symbols     提取文件符号列表
find_references 查找符号引用 (启发式)
run_tests       运行测试 (默认禁用，安全风险)
```

### 4.4 Context Builder

不做预检索。只组装系统 prompt + 仓库树 + 历史 + 用户输入。

---

## 5. API 规格

```
POST /api/chat          SSE stream (thinking/tool_result/text/done events)
POST /api/index         触发/重建索引
GET  /api/index/status
GET  /api/health        GPU/LLM/indexing 状态
GET  /api/workspace/tree?path=.
GET  /api/workspace/file?path=<path>
```

无 /api/complete (不做补全)。文件读取用 GET，与 tree 统一风格。

### 5.1 配置文件 (config.yaml)

```yaml
server:
  host: "127.0.0.1"            # 默认只绑本地回环
  port: 8080

llm:
  backend: "vllm"
  model: "Qwen/Qwen2.5-Coder-14B-Instruct"
  dtype: "float16"
  max_model_len: 16384         # 默认 16K (保守); FP8 验证通过后可提 32K
  kv_cache_dtype: "fp16"       # 默认 FP16; 应用层验证 FP8 后可切换
  gpu_memory_utilization: 0.88 # 48*0.88≈42GB: 权重28 + KV池12 + 运行时2
  enforce_eager: true          # gfx1100 HIP graph 兼容性差，默认 eager; start_llm.sh 用 ENFORCE_EAGER 环境变量控制
  vllm_version: "0.16.0"        # M0 验证后 pin 具体版本到 requirements.txt

embedding:
  model: "BAAI/bge-m3"
  device: "cpu"
  batch_size: 16

reranker:
  model: "BAAI/bge-reranker-v2-m3"
  device: "cuda"               # 在线查询关键路径
  candidate_count: 20
  final_count: 15

index:
  vector_store: "chromadb"
  persist_dir: ".raa/index"
  supported_extensions: [".py"]
  exclude_dirs: [".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".raa"]

agent:
  max_iterations: 8
  context_budget: 16384        # FP8 验证通过后可提 32768
  temperature: 0.3
  top_p: 0.9

session:
  persist: true
  persist_dir: ".raa/sessions"

security:
  run_tests_enabled: false
  test_commands: ["pytest", "python -m pytest"]
  max_test_timeout: 60

# ROCm 稳定性环境变量 (gfx1100 常见问题)
rocm_env:
  PYTORCH_ROCM_ARCH: "gfx1100"
  HSA_ENABLE_SDMA: "0"         # 修复 RDNA3 内存传输问题
  AMD_SERIALIZE_KERNEL: "3"    # 修复 kernel 竞态
  AMD_SERIALIZE_COPY: "3"      # 修复 copy 竞态
```

> **`gpu_memory_utilization: 0.88` 的理由**: 14B FP16 权重 ~28GB。
> 0.88 × 48 = 42.2GB: 权重 28 + 运行时 2 = 30GB，剩余 ~12GB 给 KV cache 池。
> 不能用 0.62: 0.62 × 48 = 29.8GB < 30GB (权重+运行时)，vLLM 会 OOM。
> Reranker 在独立进程，占用 ~1.5GB (0.6GB 模型 + 0.9GB HIP context)，
> 与 vLLM 的 0.88 上限互不干扰 (两进程各自管理 GPU 内存)。

> **`enforce_eager: true` 的理由**: gfx1100 上 HIP graph capture 经常崩溃
> 或输出异常。`--enforce-eager` 跳过 graph capture，代价是推理慢 10-20%。
> M0 验证时先试不加 (graph 模式快)，崩溃或输出异常则加。

---

## 6. 模型策略与本地部署

### 6.1 LLM 选型

| 模型 | 用途 | FP16 权重 | Q4_K_M 权重 | 备注 |
|---|---|---|---|---|
| Qwen2.5-Coder-14B-Instruct | Agent | **~28GB** | ~9GB | 主力 |
| Qwen2.5-Coder-7B-Instruct | 备选 | ~15GB | ~5GB | 14B 表现差时降级 |

> **权重大小说明**: 14B 模型有 ~14.7B 参数。FP16 每参数 2 字节:
> 14.7 × 10^9 × 2 = ~28GB。磁盘下载大小 = VRAM 占用 (非量化模型原样加载)。

### 6.2 Embedding + Reranker

| 模型 | 用途 | 大小 | 运行位置 | 理由 |
|---|---|---|---|---|
| BGE-m3 | 向量检索 | ~2.4GB | **CPU** | 索引一次性操作 |
| bge-reranker-v2-m3 | 重排序 | ~0.6GB | **GPU** | 在线查询关键路径 |

> Reranker 进程实际 GPU 占用 ~1.5GB (模型 0.6 + HIP context + allocator + PyTorch CUDA 运行时)。

### 6.3 推理后端

**主力 vLLM (ROCm 7.2.1)**:

安装 (ROCm 版没有通用 PyPI wheel):

```bash
# 方式一: ROCm 专用 index
pip install vllm==<M0验证版本> --extra-index-url https://download.pytorch.org/whl/rocm7.2.1

# 方式二: 源码编译
git clone https://github.com/vllm-project/vllm
cd vllm
pip install -e . --rocm-version=7.2.1
```

> **必须 pin 版本**: vLLM ROCm 支持变化极快，某版修了 gfx1100 下版可能 break。
> M0 验证通过的版本立即 pin 到 requirements.txt。

启动 (关键 flags):

```bash
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

# --enforce-eager: gfx1100 默认加; M0 验证 graph 后可通过 ENFORCE_EAGER=false 关闭
# --tool-call-parser hermes: Qwen2.5 function calling (旧版 vLLM 退回 hermes)
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
  --dtype float16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.88 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  --port 8000
```

> Flags 说明:
> - `--enable-auto-tool-choice --tool-call-parser hermes`: Qwen2.5 function calling 必需
> - `--enforce-eager`: gfx1100 HIP graph 兼容性差，默认跳过。M0 先试不加，崩溃则加
> - `--gpu-memory-utilization 0.88`: 权重 28GB 需要，不能用更低

**备选 llama.cpp (HIP)**:

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j
./build/bin/llama-server -m Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf -ngl 999 -c 16384 --port 8000
```

> **llama.cpp 路线的重大限制**: llama.cpp server **不支持 `--tool-call-parser`**。
> 切到 llama.cpp 意味着:
> 1. 失去结构化 tool_calls JSON 输出
> 2. Agent 完全依赖文本格式解析 (`<tool_call>...</tool_call>` 正则提取)
> 3. 14B 文本格式 tool call 比 vLLM 结构化格式更不稳定
> 4. M3 质量评估需基于文本解析，可能需要更多 prompt 工程
> 5. M3 质量门禁可能需要降低标准

### 6.4 GPU 显存分配

```
进程 1: vLLM server
  gpu_memory_utilization: 0.88 → 预分配上限 42.2GB
  内含:
    模型权重 (FP16):    ~28 GB
    PyTorch 运行时:      ~2 GB
    KV cache 池 (剩余): ~12 GB
      (支持 ~4 个并发 16K 序列，或 ~2 个并发 32K 序列)
      (单序列 16K FP16 KV cache ≈ 3.0GB，见 6.6 计算)

进程 2: FastAPI (含 reranker)
  Reranker 模型:    ~0.6 GB
  HIP context:      ~0.9 GB
  合计:             ~1.5 GB

CPU: BGE-m3 embedding (索引时)
─────────────────────────────────
GPU 总占用:       ~43.7 GB
剩余:             ~4.3 GB
```

> vLLM 和 reranker 在不同进程，各自管理 GPU 内存，互不感知。
> vLLM 只看到自己的 0.88 上限。reranker 在 vLLM 预分配之外的空间运行。

### 6.5 显存预算汇总

**vLLM 路线 (FP16 权重 + FP16 KV cache)**:

```
模型权重 (14B FP16):     ~28 GB
KV cache 池 (vLLM 分配): ~12 GB
PyTorch 运行时:           ~2 GB
Reranker 进程:           ~1.5 GB
──────────────────────────────────
GPU 总计:                ~43.7 GB / 48 GB
剩余:                    ~4.3 GB
上下文:                  max_model_len = 16384 (保守默认)
```

**如果 FP8 权重量化可用 (--quantization fp8)**:

```
模型权重 (14B FP8):      ~15 GB   ← 省 ~13GB!
KV cache 池:             ~25 GB   ← 大幅增加
PyTorch 运行时:           ~2 GB
Reranker 进程:           ~1.5 GB
──────────────────────────────────
GPU 总计:                ~43.7 GB
上下文:                  max_model_len 可提至 32768
但有 dequantization 计算开销，tokens/s 可能下降
```

> **FP8 权重量化 vs FP8 KV cache 是两件不同的事**:
> - `--quantization fp8`: 权重从 FP16 压到 FP8，省 ~13GB 显存。省的是大头。
> - `--kv-cache-dtype fp8`: KV cache 从 FP16 压到 FP8，省 ~6GB (32K)。
> 文档之前只纠结后者，完全漏了前者。
> **如果目标是让 14B 舒服跑 32K，应优先验证 `--quantization fp8`，不是 FP8 KV cache。**

**llama.cpp 路线 (Q4_K_M)**:

```
模型权重 (Q4_K_M):       ~9 GB
KV cache (Q8, 32K):      ~3 GB   (单序列)
HIP 运行时:               ~1-2 GB
Reranker (GPU):          ~1.5 GB
──────────────────────────────────
GPU 总计:                ~14-15 GB / 48 GB
剩余:                    ~33 GB (非常宽裕)
```

### 6.6 KV cache 大小计算

> Qwen2.5-Coder-14B: 48 层, 8 KV heads (GQA), head_dim=128

```
每 token KV cache (FP16):
  2 (K+V) × 48 层 × 8 heads × 128 dim × 2 bytes = 196,608 bytes ≈ 192 KB/token

单序列 16K (FP16): 192KB × 16384 ≈ 3.0 GB
单序列 32K (FP16): 192KB × 32768 ≈ 6.0 GB
单序列 32K (FP8):   96KB × 32768 ≈ 3.0 GB
```

> 注意区分: **单序列 KV cache** (~3GB) vs **vLLM KV cache 池** (~12GB)。
> vLLM 用剩余显存预分配一个池，支持多个并发序列。
> 单用户 localhost 产品实际只跑 1 个序列，但 vLLM 仍会预占整个池。

---

## 7. AMD Radeon GPU 推理优化

### 7.1 关键假设与验证状态

| 优化项 | 状态 | 验证 |
|---|---|---|
| ROCm 7.2.1 + PyTorch on gfx1100 | 待验证 | M0 |
| MIOpen (ROCm attention backend) on gfx1100 | **待验证** | M0 |
| vLLM 编译/运行 on gfx1100 | **高风险** | M0 提前 |
| HIP graph capture on gfx1100 | **高风险** | M0 (加 --enforce-eager 兜底) |
| FP8 权重量化 (--quantization fp8) | **大概率也不可用** | M1 |
| FP8 KV cache (--kv-cache-dtype fp8) | **大概率不可用** | M1 |
| RDNA3 GEMM TFLOP 利用率 < CDNA | 预期 (无法改变) | benchmark 标注 |
| 检索 + reranking (应用层) | 确认 | M3 |

### 7.2 FP8 的两个维度

> 之前版本混淆了两个完全不同的 FP8 优化。

**维度一: FP8 权重量化 (`--quantization fp8`)**

```
效果: 权重 28GB → ~15GB，省 ~13GB
影响: 每次推理有 dequantization 开销，tokens/s 可能下降 10-20%
适用: 显存不够时的有效手段，省的显存可用于更长上下文
状态: 大概率也不可用 (vLLM ROCm 的 FP8 kernel 同样为 gfx942 写，gfx1100 大概率无对应实现)
```

**维度二: FP8 KV cache (`--kv-cache-dtype fp8`)**

```
效果: KV cache 显存减半 (32K: 6GB→3GB/序列)
影响: attention 计算 cast 回 FP16，有类型转换开销
状态: 大概率不可用。vLLM ROCm 的 FP8 KV cache 是为 gfx942 (MI300X) 写的，
      gfx1100 很可能没有对应 kernel 实现。
      最可能结果: vLLM 直接拒绝启动报错 "FP8 KV cache not supported on this backend"。
      而不是"启动成功但速度慢"。
```

FP8 权重量化和 FP8 KV cache 在 gfx1100 上大概率都不可用——
两者依赖的 vLLM ROCm FP8 kernel 同样是为 gfx942 写的。
benchmark 叙事重心应转向 "FP16 + 检索策略"，不依赖任何 FP8 优化。
如果 M1 验证意外可用，则是 bonus (权重省 ~13GB，KV cache 省 ~6GB)。

> 如动态 `--quantization fp8` 质量不达标 (动态 FP8 做 per-tensor scaling，
> 不做 per-channel，精度通常更差)，可尝试预校准的静态 FP8 checkpoint:
> 用 `llm-compressor` 或 `AutoFP8` 做 calibration + per-channel FP8 量化，
> 产出一个预量化模型。vLLM 加载时不需要 `--quantization fp8`，直接加载该
> checkpoint 即可。静态量化精度通常比动态量化好 10-15%，但需要额外校准步骤。

### 7.3 vLLM 对 gfx1100 的支持

vLLM ROCm 官方聚焦 gfx90a/gfx942。RDNA3 碎片化。

**MIOpen**: vLLM ROCm 用 `rocm_flash_attn` backend，依赖 MIOpen。
gfx1100 上可能缺 kernel → 报错或静默 fallback 到慢实现。

**Attention backend 可切换**:

```bash
VLLM_ATTENTION_BACKEND=ROCM_FLASH_ATTN  # 默认
VLLM_ATTENTION_BACKEND=FLASHINFER        # 可尝试，ROCm 支持因版本而异
```

如果默认 backend 报错，切换是重要调试手段。

**HIP graph**: gfx1100 上 graph capture 经常崩溃或输出异常。
`--enforce-eager` 跳过，代价慢 10-20%。

**对策**: M0 提前试编译 + MIOpen check + graph mode test。

### 7.4 RDNA3 kernel 性能预期

CDNA Matrix Core GEMM 效率远高于 RDNA3 WMMA。FFN matmul 在 gfx1100 上的
TFLOP 利用率可能只有 CDNA 一半。W7900 (理论 ~45 FP16 TFLOPS) 吞吐可能不如
A10 (31 TFLOPS)。benchmark 需诚实标注。

### 7.5 GDDR6 带宽与理论吞吐上限
W7900 的 GDDR6 带宽为 864 GB/s。LLM 推理的 decode 阶段是**带宽瓶颈**
(每生成一个 token 需读取全部权重):
```
理论吞吐上限 = 带宽 / 权重大小
            = 864 GB/s / 28 GB (14B FP16)
            ≈ 31 tok/s
```
这是**理论天花板**——假设 100% 带宽利用率、零开销。
实际受 kernel 效率、attention 计算、RDNA3 GEMM 折损影响，
预期实测在 15-25 tok/s 区间。
对比: MI300X (5300 GB/s HBM3 + FP8 权重 15GB) 理论上限 ~350 tok/s，
W7900 受 GDDR6 带宽和 RDNA3 kernel 折损双重限制，~20 tok/s 是合理预期。
10.1 的 "> 15 tok/s" 目标已基于此分析调整。
这条分析应在 benchmark 文档中展示，帮助评委理解性能数字的物理边界。

### 7.6 不依赖 FP8 的优化

**检索 + reranking**: embedding top-20 (CPU) + GPU rerank top-15。
"在 48GB 上用 reranked 检索达到等效效果"是核心叙事。

**上下文裁剪**: get_symbols 看结构 → read_file 精准读取。

**llama.cpp 量化**: Q4_K_M 权重 ~9GB。

### 7.7 Benchmark 设计

| Suite | 测试内容 | 指标 | 公平性 |
|---|---|---|---|
| backend | vLLM(FP16) vs llama.cpp(Q8_0) vs llama.cpp(Q4_K_M) | tokens/s, TTFT | 同精度: vLLM FP16 vs Q8_0; Q4_K_M 单列 |
| context_length | 4K/8K/16K/32K | TTFT, tokens/s, VRAM | |
| batch | 1/4/8 并发 | 总吞吐, 平均延迟 | 注明"展示上限，非典型场景" |
| quantization | Q4_K_M/Q5_K_M/Q8_0 (llama.cpp) | tokens/s, VRAM, 质量 | |
| fp8_weight | FP8 权重 vs FP16 权重 | VRAM, tokens/s, 质量 | 大概率不可用，记录失败也是数据 |
| fp8_kv | (仅当可用) FP8 vs FP16 KV | VRAM, tokens/s | 大概率不可用，记录失败也是数据 |
| rag_vs_longctx | 检索+rerank+16K vs 纯32K | 质量 + 速度 | 基线: 目标文件+同目录+import 链塞满 |
| rerank_impact | 有/无 reranker | 质量通过率 | |
| eager_vs_graph | eager vs graph 模式 | tokens/s | gfx1100 稳定性数据 |

---

## 8. 会话与状态管理

JSON 持久化到 `.raa/sessions/`。重启恢复最近 10 个会话。

---

## 9. 安全模型

### 9.1 网络绑定
API 默认 127.0.0.1。远程访问 opt-in。

### 9.2 命令执行
pytest collection 阶段 import conftest.py = **任意代码执行**。
run_tests 默认禁用，需手动开启。只在信任仓库使用。

### 9.3 文件访问
限仓库根目录内。敏感文件不索引。Agent 只读。

### 9.4 隐私
全本地，无遥测。tcpdump 可验证零出站。

---

## 10. 非功能性需求

### 10.1 性能指标 (待实测)

| 指标 | 目标 |
|---|---|
| Agent 首 token 延迟 | < 3s (16K, 含 reranker GPU) |
| Agent 生成速度 | > 15 tok/s (理论上限 ~31，见 7.5) |
| Reranker 延迟 (GPU) | < 0.5s (20 candidates) |
| 仓库索引速度 (CPU) | ~6 文件/min |

### 10.2 可靠性
重连/落盘/续索引/自动降级。

### 10.3 Agent 质量评估
20 题 + LLM-as-judge 半自动。目标 > 70% 达 2 分。

---

## 11-12. 测试与目录结构

(同 v3.1，无变化)

---

## 13. 评委运行体验 (README)

```
1. 环境: ROCm 7.2.1.x + Python 3.10+
2. ./download_models.sh (14B ~28GB + BGE-m3 ~2.4GB + reranker ~0.6GB)
3. pip install -r requirements.txt (vllm 版本已 pin)
4. bash start_llm.sh (含 tool_call flags + enforce_eager + ROCm env)
5. python src/server.py
6. 冷启动: vLLM 加载 14B FP16 ~2-4min (28GB 权重从磁盘到显存) | 索引 50 文件 (CPU) ~8min | 首答 ~3-5s
7. 离线: 模型预下载 + 预索引示例仓库
```

---

## 14. 技术栈

Python 3.10 / vLLM (ROCm 7.2.1, 版本 pin) / llama.cpp (HIP) / ROCm+MIOpen /
FastAPI / tree-sitter / BGE-m3 (CPU) / bge-reranker (GPU) / ChromaDB / ripgrep
