# RepoAgent — 实施计划书

> 版本: 3.2 | 日期: 2026-08-04
> 关联文档: spec.md v3.2
> 预计工期: 16-22 天 (基准 × 1.5 缓冲)

---

## 1. 里程碑总览

| 里程碑 | 名称 | 基准 | 含缓冲 | 质量门禁 |
|---|---|---|---|---|
| M0 | 环境 + vLLM 预编译 + MIOpen + graph | 1.5 天 | 2.5 天 | PyTorch/MIOpen 可用 + vLLM 路线 + graph 模式结论 |
| M1 | 推理 + FP8 决策 + tool_call | 2 天 | 3 天 | 14B 推理通过 + FP8 权重结论 + tool_call 验证 |
| M2 | 索引 + reranking | 2 天 | 3 天 | Python 索引 + rerank 可用 |
| M3 | Agent + 质量评估 | 3 天 | 5 天 | 20 题 LLM-judge > 70% |
| M4 | Web UI | 1.5 天 | 2.5 天 | 对话 + 思考可视化 |
| M5 | AMD benchmark | 1 天 | 2 天 | 4 组数据 |
| M6 | 交付物 | 1.5 天 | 2.5 天 | 视频 + PPT |

**总基准: 12.5 天 | 含缓冲: 20.5 天**

```
Day 1-2     Day 3-5     Day 6-8    Day 9-13     Day 14-16   Day 17-20
M0 ---------+-----------+----------+------------+-----------+
            M1          M2         M3           M4+M5      M6
```

---

## 2. 各里程碑详细计划

### M0: 环境 + vLLM 预编译 + MIOpen + graph (Day 1-2)

**目标**: 确认 ROCm + PyTorch + MIOpen + vLLM 编译 + HIP graph 兼容性。

| # | 任务 | 验证方法 | 基准 | 缓冲 |
|---|---|---|---|---|
| 0.1 | ROCm 驱动 | rocminfo grep gfx = gfx1100 | 10min | |
| 0.2 | HIP 工具链 | hipconfig --version | 5min | |
| 0.3 | 安装 ROCm 版 PyTorch | pip install torch --index-url rocm6.2 | 30min | +1h |
| 0.4 | PyTorch GPU 验证 | torch.cuda.is_available() = True | 5min | |
| 0.5 | 确认 GPU 型号 | get_device_name = W7900 | 5min | |
| 0.6 | **锁定 ROCm 版本号** | 记录 6.2.x 具体版本 | 5min | |
| 0.7 | **MIOpen 可用性检查** | 跑小型 attention 操作 | 30min | +1h |
| 0.8 | **设置 ROCm 稳定性 env** | HSA_ENABLE_SDMA=0 等 | 5min | |
| 0.9 | **提前试 vLLM 编译** | pip install 或源码编译 | 1h | +3h |
| 0.10 | **HIP graph 兼容性测试** | 加载小模型试 graph mode | 30min | +1h |
| 0.11 | (vLLM 失败时) llama.cpp HIP 编译 | cmake GGML_HIP | 1h | +1h |

**M0 关键决策产出**:

```
vLLM 编译成功?
  +- 是 -> M0.10: HIP graph 能用?
  |         +- 是 -> M1 走 vLLM，graph 模式 (快)
  |         +- 否 -> M1 走 vLLM，强制 --enforce-eager (慢 10-20% 但稳定)
  +- 否 -> 切 llama.cpp，M0 额外完成编译验证
```

**ROCm 稳定性环境变量**:

```bash
export PYTORCH_ROCM_ARCH=gfx1100     # 编译目标 (如需源码编译)
export HSA_ENABLE_SDMA=0             # 修复 RDNA3 内存传输
export AMD_SERIALIZE_KERNEL=3        # 修复 kernel 竞态
export AMD_SERIALIZE_COPY=3          # 修复 copy 竞态
```

> 这些不是冷门配置。gfx1100 相关稳定性问题一半以上的 solution 涉及它们。
> verify_rocm.py 应包含一轮推理稳定性测试 (连续跑 5 次，检查输出一致性)。

> **关于 HSA_OVERRIDE_GFX_VERSION**: W7900 的原生代号就是 gfx1100。
> override 成 `11.0.0` (等于自己) 通常无意义。
> 如果遇到不被识别，正确做法是编译时 `PYTORCH_ROCM_ARCH=gfx1100`。

**潜在问题**:

| 问题 | 正确解决 |
|---|---|
| torch.cuda.is_available() = False | ROCm 版本与 wheel 不匹配 |
| MIOpen 缺 gfx1100 kernel | 记录日志，严重则切 llama.cpp |
| HIP graph 崩溃/输出异常 | 加 `--enforce-eager` |
| vLLM 编译报错 | 记录，切 llama.cpp |
| 推理偶发 NaN/结果不一致 | 设置 AMD_SERIALIZE_KERNEL=3 |

**质量门禁**: verify_rocm.py 全 [OK]，含 MIOpen check + 稳定性 check。

---

### M1: 推理后端 + FP8 决策 + tool_call (Day 3-5)

**目标**: LLM 推理可用 + FP8 权重量化结论 + tool_call 验证。

#### 1A: 推理服务启动

**vLLM 路线**:

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 1.1 | 下载 Qwen2.5-Coder-14B-Instruct (~28GB) | 1h | +1h |
| 1.2 | 启动 vllm serve | 10min | |
| 1.3 | 验证短上下文推理 | 10min | |

启动命令:

```bash
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
  --dtype float16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.88 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen2 \
  --enforce-eager \
  --port 8000
```

> `--enforce-eager` 默认加。M0 如果 graph mode 可用，去掉此 flag 提速 10-20%。
> `--gpu-memory-utilization 0.88`: 14B FP16 权重 ~28GB，必须用 0.88 (42GB)。
> 不能用 0.62: 29.8GB < 30GB (权重+运行时)，vLLM 会 OOM。

> **必须 pin vLLM 版本**: M0 验证通过的版本立即写入 requirements.txt
> (`vllm==0.6.x`)。ROCm 支持变化快，不 pin 下次 install 可能 break。

**llama.cpp 路线**:

| # | 任务 | 基准 |
|---|---|---|
| 1.1b | 下载 GGUF (Q4_K_M) | 20min |
| 1.2b | 启动 llama-server | 5min |
| 1.3b | 验证 /completion | 10min |

> **llama.cpp 路线的重大限制**: 不支持 `--tool-call-parser`。
> 失去结构化 tool_calls，Agent 完全依赖文本格式解析。
> M3 质量评估需基于文本解析，tool_call 可靠性下降。

#### 1B: FP8 权重量化验证 (优先)

> 验证 FP8 权重量化 (大概率不可用，同 FP8 KV cache 原因)。不是优先项而是确认项。

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 1.4 | 尝试 `--quantization fp8` 启动 | 15min | +30min |
| 1.5 | **smoke test** (验证客观指标) | 15min | +30min |
| 1.6 | 对比 FP8 vs FP16 显存和速度 | 30min | |

**FP8 权重验证流程**:

```
尝试 --quantization fp8
  |
  +-- 启动失败
  |     -> 记录错误，FP16 权重，max_model_len=16384
  |
  +-- 启动成功
        -> 跑 smoke test:
           a. 发送已知 prompt ("写一个 Python hello world")
           b. 检查输出: 不含 NaN/Inf、长度 > 10 字符、连跑 5 次输出一致
           c. 记录 tokens/s
           d. 连跑 5 次检查一致性 (排除偶发溢出)
        |
        +-- smoke test 通过 + 速度可接受
        |     -> 使用 FP8 权重，max_model_len 可提至 32768
        |
        +-- smoke test 失败 (含 NaN/Inf、长度为 0、或 5 次输出不一致)
              -> 回退 FP16 权重，max_model_len=16384
```

> **失败不止是启动异常**。四种失败模式:
> a) 启动报错 — catch 能抓
> b) 启动成功但输出异常 (含 NaN/Inf、长度为 0) — 没有 exception
> c) 启动成功但极慢 — 没有 exception
> d) 偶发数值溢出 — 测试通过但生产出问题
> **smoke test 是必须的，不能只 catch 异常。**

#### 1C: FP8 KV cache 验证 (大概率失败)

| # | 任务 | 基准 |
|---|---|---|
| 1.7 | 尝试 `--kv-cache-dtype fp8` | 15min |

> vLLM ROCm 的 FP8 KV cache 是为 gfx942 (MI300X) 写的。
> gfx1100 最可能结果: 直接拒绝启动报错。
> 不是"启动成功但慢"——是"根本没这个 kernel"。
> 记录失败也是有效数据 (benchmark 标注"不支持")。

#### 1D: tool_call 验证

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 1.8 | 发送带 tools 参数请求 | 10min | |
| 1.9 | 验证结构化 tool_calls 返回 | 10min | +30min |

> 加了 `--tool-call-parser qwen2` 但 tool_calls 仍空?
> 检查 Qwen2.5 chat template 匹配。试 hermes parser (旧版 vLLM 不支持 qwen2)。
> 最后退路: 文本格式解析 (正则提取)。

#### 1E: 推理封装层

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 1.10 | backend.py 抽象接口 | 1h | |
| 1.11 | VLLMBackend 或 LlamaCppBackend | 2h | +1h |
| 1.12 | **FP8 try-fail-fallback + smoke test** | 2h | +1h |
| 1.13 | tool_parser.py (双格式) | 2h | +1.5h |

> backend.py 初始化:
> 1. 尝试 `--quantization fp8` 启动
> 2. catch 异常 + 跑 smoke test (客观指标: NaN/Inf、长度、一致性)
> 3. 失败则 kill + `--dtype float16` 重启
> 这个逻辑是应用层职责。

**质量门禁**:
- 推理正常返回
- FP8 权重结论已定 (用/不用)
- tool_call 返回结构化 JSON
- FP8 fallback + smoke test 工作

---

### M2: 代码索引层 (Day 6-8)

> Embedding 跑 CPU，Reranker 跑 GPU。

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 2.1 | tree-sitter + Python grammar 安装 | 1h | +1h |
| 2.2 | parser.py (AST chunk 提取) | 4h | +2h |
| 2.3 | get_symbols 接口 | 2h | |
| 2.4 | embeddings.py (CPU batch) | 1.5h | |
| 2.5 | reranker.py (GPU cross-encoder) | 2h | +1h |
| 2.6 | ChromaDB + 持久化 | 30min | |
| 2.7 | indexer.py (完整流程) | 3h | +2h |
| 2.8 | 增量索引 (SHA256) | 1.5h | |
| 2.9 | 符号表 + find_references | 3.5h | +1h |
| 2.10 | 验证: embedding(CPU) -> rerank(GPU) -> top-15 | 30min | |

**质量门禁**: 50+ 文件 Python 索引完成。reranker GPU < 0.5s。增量索引正确。

---

### M3: Agent 核心 + 质量评估 (Day 9-13)

> **最大风险。**

#### 3A: 工具实现

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 3.1 | search.py (含 GPU rerank) + grep_code | 2.5h | +1h |
| 3.2 | files.py | 1.5h | |
| 3.3 | ast_tools.py | 1.5h | +1h |
| 3.4 | exec.py (默认禁用) | 1h | |
| 3.5 | registry.py | 1.5h | |

#### 3B: ReAct 引擎

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 3.6 | engine.py 主循环 (max 8 轮) | 3h | +2h |
| 3.7 | prompts.py | 1.5h | |
| 3.8 | context.py (双预算 16K/32K) | 3h | +1h |
| 3.9 | tool_parser.py (双格式) | 2h | +2h |
| 3.10 | 端到端测试 | 1h | +2h |

> **llama.cpp 路线注意**: 如果走 llama.cpp，tool_call 只有文本格式。
> M3 质量评估必须基于文本解析跑。预期通过率低于 vLLM 路线。

#### 3C: Agent 质量评估

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 3.11 | 准备固定示例仓库 (Flask) | 1h | |
| 3.12 | 编写 20 题测试集 | 2h | +1h |
| 3.13 | grade.py (LLM-as-judge) | 2h | +1h |
| 3.14 | 第一次评估 | 1h | |
| 3.15 | 调优 prompt (可能 3-5 轮) | 4h | +4h |
| 3.16 | 第二次评估 | 1h | |
| 3.17 | 人工复核分歧项 | 1h | |

**评估流程**:

```
1. eval_agent.py 遍历 20 题
2. grade.py LLM judge 评 0-3 分
3. 分歧项标记人工复核
4. 统计通过率
迭代: 只跑 1-2 (自动)
正式: 跑完整 1-4
目标: > 70% 达 2 分
```

> 通过率 < 50% 时检查:
> - tool_call 解析正确否
> - search_code 返回相关否
> - 加回预检索混合模式
> - 换 7B 模型

**质量门禁**: 20 题 > 70% (vLLM) / > 60% (llama.cpp，标准降低)。

---

### M4: Web UI (Day 14-16)

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 4.1 | routes.py + schemas.py | 3.5h | |
| 4.2 | SSE 流式 | 2h | +1h |
| 4.3 | index.html + styles.css | 3.5h | +1h |
| 4.4 | app.js (SSE + 渲染 + 思考展示) | 4h | +2h |
| 4.5 | highlight.js | 1h | |

---

### M5: AMD Benchmark (Day 17-18)

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 5.1 | benchmark.py | 3h | +1h |
| 5.2 | 后端对比 (同精度) | 30min | |
| 5.3 | 上下文长度影响 | 30min | |
| 5.4 | batch 吞吐 (叙事说明) | 30min | |
| 5.5 | 量化对比 | 30min | |
| 5.6 | **FP8 权重 vs FP16 权重** | 30min | |
| 5.7 | FP8 KV (记录不可用也是数据) | 15min | |
| 5.8 | 检索+rerank vs 纯长上下文 | 1h | |
| 5.9 | reranker 影响 | 30min | |
| 5.10 | **eager vs graph 模式** | 30min | |
| 5.11 | amd_optimization.md | 1.5h | +1h |

**公平性要求**:
- 后端对比: vLLM FP16 vs llama.cpp Q8_0 (同精度)
- 纯长上下文基线: 目标文件+同目录+import 链塞满，不是随机
- batch: 注明"展示上限，非典型场景"

---

### M6: 交付物 (Day 19-20)

| # | 任务 | 基准 | 缓冲 |
|---|---|---|---|
| 6.1 | 架构图 | 2h | |
| 6.2 | README (含评委 Quick Start) | 2h | +1h |
| 6.3 | 代码清理 | 1.5h | +1h |
| 6.4 | download_models.sh + start_llm.sh | 1h | |
| 6.5 | Demo 视频 | 2.5h | +1h |
| 6.6 | PPT | 2.5h | |

**start_llm.sh 内容**:

```bash
#!/bin/bash
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

# enforce_eager 用环境变量控制，M0 验证 graph mode 后切换
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"   # M0 验证 graph 可用后改为 false
EAGER_FLAG=""
if [ "$ENFORCE_EAGER" = "true" ]; then
  EAGER_FLAG="--enforce-eager"
fi

# FP8 权重量化大概率不可用，如意外可用则 QUANT=fp8
QUANT="${QUANT:-none}"                   # none | fp8
QUANT_FLAG=""
MAX_LEN=16384
if [ "$QUANT" = "fp8" ]; then
  QUANT_FLAG="--quantization fp8"
  MAX_LEN=32768
fi

vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
  --dtype float16 \
  --max-model-len $MAX_LEN \
  --gpu-memory-utilization 0.88 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen2 \
  $EAGER_FLAG \
  $QUANT_FLAG \
  --port 8000
# ENFORCE_EAGER=false 提速 10-20% (需 M0 验证 graph mode 稳定)
# QUANT=fp8 省 ~13GB 显存 + 提上下文 (需 M1 验证，大概率不可用)
```

#### Demo 视频 (3-5 分钟)

```
[0:00-0:30] 痛点
[0:30-1:00] 架构 (全本地 + W7900)
[1:00-3:00] Demo (索引 -> 提问 -> Agent 检索 -> 回答)
[3:00-4:00] AMD 数据
  主线: 检索+rerank vs 纯长上下文, GDDR6 带宽理论分析, RDNA3 折损诚实标注
  (如 FP8 可用则加 FP8 对比; 不可用则展示 "gfx1100 不支持 FP8" 的诚实结论)
[4:00-4:30] 总结
```

#### README 评委体验清单

```
- [ ] 环境 (ROCm 6.2.x + Python 3.11+)
- [ ] download_models.sh (14B ~28GB)
- [ ] start_llm.sh (含 tool_call + enforce_eager + ROCm env)
- [ ] requirements.txt (vllm 版本 pin)
- [ ] 预索引示例仓库
- [ ] 冷启动时间预估 (vLLM 14B ~2-4min, 索引 50文件 ~8min, 首答 ~3-5s)
- [ ] 离线场景说明
- [ ] troubleshooting (ROCm env vars, graph mode, FP8)
```

---

## 3. 风险矩阵

| 风险 | 概率 | 影响 | 预防 | 应急 |
|---|---|---|---|---|
| **vLLM 不支持 gfx1100** | **高** | **高** | M0 提前编译 | 切 llama.cpp |
| **MIOpen 缺 gfx1100 kernel** | **中** | **高** | M0 attention check | 切 llama.cpp |
| **HIP graph 崩溃** | **高** | **中** | M0 测试 graph mode | --enforce-eager (慢但稳) |
| **FP8 权重量化大概率不可用** | **高** | **低** | M1 验证 (预期失败) | FP16 权重 + 16K (预期内的默认方案) |
| **FP8 KV cache 不支持** | **高** | **低** | M1 记录失败 | FP16 KV (预期内的回退) |
| **Agent 质量差** | **高** | **高** | M3 跑 20 题 | 换 7B；混合模式；减工具 |
| **llama.cpp tool_call 不可靠** | 中(vLLM) / 高(llama.cpp) | 高 | M1 优先保 vLLM | 加强文本解析 + 更多 prompt 工程 |
| **14B 太慢 (RDNA3 kernel)** | 中 | 中 | benchmark 提前跑 | 降级 7B |
| vLLM 版本 break | 中 | 中 | pin 版本 | 回退到验证通过版本 |

### 风险优先级

> **M0 三连**: vLLM 编译 + MIOpen + HIP graph。
> 任一失败都可能迫使切 llama.cpp (丢结构化 tool_call)。
>
> **M1 FP8**: 权重量化是 bonus (省显存)，不可用就 FP16+16K，不影响核心功能。
> FP8 KV cache 预期不可用，记录失败即可。
>
> **M3 Agent**: 最大产品风险。llama.cpp 路线下更严重 (文本解析不可靠)。

---

## 4. MVP 定义

**必须完成**: M0+M1 推理 / M2 索引+rerank / M3 Agent+4工具+20题 / M4 UI / M5 3组benchmark / M6 全套

**可砍**: find_references / run_tests / 增量索引 / 精美前端

**不可砍**: 20 题评估 / GPU reranking / 思考可视化

---

## 5. 依赖关系

```
M0 (环境 + vLLM + MIOpen + graph)
 -> M1 (推理 + FP8 + tool_call) -> M5 (Benchmark)
 -> M2 (索引 + rerank)
 -> M3 (Agent + 评估)
 -> M4 (UI) -> M6 (交付)
```

---

## 6-9. 规范与检查清单

(同 v3.1 结构，内容已更新)

### 最终质量检查清单

**功能**: Agent 回答/定位/读取多文件 | 20 题 > 70% | 索引+rerank | UI
**AMD**: vLLM/llama.cpp 运行 | FP8 权重结论 | MIOpen 验证 | graph 结论 | 3+ benchmark | RDNA3 折损标注
**隐私**: 127.0.0.1 | 无外部请求 | 无遥测
**交付**: README+Quick Start | download/start 脚本 | vllm pin | 架构图 | benchmark | 视频 | PPT
