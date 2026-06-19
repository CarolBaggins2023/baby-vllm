# baby-vllm

[English](README.md) | 简体中文

baby-vllm 是一个基于 nano-vllm 的轻量级、学习导向的大模型推理项目。这个仓库刻意保持小巧，重点是让分页 KV cache 管理、调度、chunked prefill 和在线请求处理等核心机制更容易阅读、修改和理解。项目目标是在一个紧凑的代码库中，把若干重要的推理服务思想讲清楚、跑起来、方便实验。

## 亮点

- 基于简单的 nano-vllm baseline 构建。
- 增加 scheduler-level continuous batching：prefill 和 decode 可以在调度层一起调度，但在 model runner 层仍然作为独立调用执行。
- 增加 chunked prefill，避免长 prompt 长时间独占整个 engine step。
- 增加在线 HTTP 服务，用于基础的流式推理实验。
- 增加单机张量并行（TP）和数据并行（DP），用于本地多 GPU 的离线和在线推理。

## 文档

- [推理引擎流程图](docs/flowchart/README.md)：用 Mermaid 图梳理离线/在线推理、调度器、KV cache、model runner、attention、并行与 Qwen3 模型等关键路径。

## 分支

本仓库中的分支也记录了项目实现和学习的推进顺序。

| 分支 | 说明 |
| --- | --- |
| `basic` | baseline 分支，基本等价于 nano-vllm。 |
| `feature-ChunkedPrefill` | 在 `basic` 基础上增加 continuous batching 和 chunked prefill。 |
| `feature-BatchReorder` | 在 `feature-ChunkedPrefill` 基础上实验 batch reordering；这个想法后来在最终版本中放弃。 |
| `basic-online` | 在 `basic` 基础上增加在线推理。 |
| `basic-dp` | 在 `basic` 基础上增加 DP 和 TP。 |
| `basic-online-dp` | 在 `basic-online` 基础上增加 DP 和 TP。 |
| `feature-Online` | 在 `feature-ChunkedPrefill` 基础上增加在线推理。 |
| `feature-online-dp` | 在 `feature-Online` 基础上增加 DP 和 TP。 |
| `main` | 当前与 `feature-online-dp` 是同一条工作线。 |

## 模型支持

baby-vllm 目前内置只支持 Qwen3 模型。使用者可以像 nano-vllm 一样自行扩展需要的模型：添加对应的模型实现，并在 model runner 中接入。下面的 benchmark 结果使用 Qwen3 0.6B、Qwen3 4B 和 Qwen3 8B 采集。

## 在线 Benchmark 结果

硬件：NVIDIA RTX 4090D。

场景：HTTP streaming 模式，64 个请求，Poisson 到达，8 个并发 client，混合 short/long prompts，`max_model_len=8192`，`max_num_batched_tokens=16384`，`max_num_sequences=64`，`max_prefill_tokens_per_step=8192`，`max_prefill_chunk_size=2048`。

| 模型 | Total Tokens | Wall Time (s) | Throughput (tok/s) | RPS | Avg TTFT (ms) | P90 TTFT (ms) | Avg TPOT (ms/token) | P90 TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B | 206,099 | 133.65 | 1,542.03 | 0.479 | 43.68 | 38.96 | 6.47 | 7.08 | 15.29 | 25.23 | 73.39 |
| Qwen3 4B | 206,099 | 311.86 | 660.87 | 0.205 | 92.11 | 112.20 | 15.48 | 16.13 | 36.30 | 59.45 | 87.93 |
| Qwen3 8B | 206,099 | 483.50 | 426.27 | 0.132 | 1,419.05 | 2,895.37 | 23.59 | 24.80 | 56.51 | 89.53 | 91.94 |

## 并行在线 Benchmark 结果

硬件：NVIDIA RTX 4090D。

场景：与上面的单 GPU 在线 benchmark 相同，HTTP streaming 模式，64 个请求，Poisson 到达，8 个并发 client，混合 short/long prompts，`max_model_len=8192`，`max_num_batched_tokens=16384`，`max_num_sequences=64`，`max_prefill_tokens_per_step=8192`，`max_prefill_chunk_size=2048`。Speedup 基于上一张表中相同模型的单 GPU 结果计算。

| 模型 | DP | TP | Throughput (tok/s) | Speedup vs 1 GPU | RPS | Avg TTFT (ms) | Avg TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B | 2 | 1 | 1,643.70 | 1.07x | 0.510 | 84.69 | 6.00 | 14.10 | 22.36 | 59.17 |
| Qwen3 0.6B | 1 | 2 | 1,575.30 | 1.02x | 0.489 | 103.59 | 6.35 | 14.97 | 24.57 | 74.70 |
| Qwen3 0.6B | 2 | 2 | 1,555.75 | 1.01x | 0.483 | 107.78 | 6.38 | 15.06 | 24.70 | 52.87 |
| Qwen3 4B | 2 | 1 | 706.10 | 1.07x | 0.219 | 113.50 | 14.39 | 33.71 | 54.45 | 82.44 |
| Qwen3 4B | 1 | 2 | 839.51 | 1.27x | 0.261 | 142.75 | 12.16 | 28.57 | 46.96 | 81.99 |
| Qwen3 4B | 2 | 2 | 906.82 | 1.37x | 0.282 | 140.97 | 11.28 | 26.47 | 42.30 | 72.72 |
| Qwen3 8B | 2 | 1 | 476.30 | 1.12x | 0.148 | 149.68 | 21.52 | 50.39 | 81.68 | 90.31 |
| Qwen3 8B | 1 | 2 | 624.82 | 1.47x | 0.194 | 187.06 | 16.37 | 38.40 | 62.43 | 86.86 |
| Qwen3 8B | 2 | 2 | 668.22 | 1.57x | 0.208 | 180.53 | 15.32 | 35.90 | 56.74 | 81.14 |

## Qwen3 8B 高压在线 Benchmark 结果

硬件：NVIDIA RTX 4090D。

场景：Qwen3 8B，HTTP streaming 模式，128 个请求，Poisson 到达，目标请求率为 1.0 requests/s，16 个并发 client，混合 short/long prompts，`max_model_len=8192`，`max_num_batched_tokens=16384`，`max_num_sequences=128`，`max_prefill_tokens_per_step=8192`，`max_prefill_chunk_size=2048`。Speedup 基于高压场景下的 `DP=1, TP=1` baseline 计算。

| 模型 | DP | TP | Throughput (tok/s) | Speedup vs DP1TP1 | RPS | Avg TTFT (ms) | P90 TTFT (ms) | Avg TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 8B | 1 | 1 | 454.96 | 1.00x | 0.135 | 35,335.35 | 66,050.15 | 31.34 | 107.99 | 144.53 | 91.27 |
| Qwen3 8B | 2 | 1 | 764.05 | 1.68x | 0.227 | 2,499.55 | 8,195.31 | 25.31 | 63.51 | 96.78 | 82.89 |
| Qwen3 8B | 1 | 2 | 925.98 | 2.04x | 0.275 | 204.21 | 240.15 | 21.93 | 53.31 | 84.40 | 85.57 |
| Qwen3 8B | 2 | 2 | 1,065.36 | 2.34x | 0.316 | 190.27 | 243.02 | 18.64 | 45.39 | 69.57 | 73.30 |

## 快速开始

以 editable 模式安装：

```bash
pip install -e .
```

将 `MODEL` 设置为本地 Qwen3 模型路径或 Hugging Face 模型标识：

```bash
export MODEL=/path/to/Qwen3
```

在一个终端中启动服务：

```bash
python -m babyvllm.entrypoints.cli \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-sequences 64 \
  --max-prefill-tokens-per-step 8192 \
  --max-prefill-chunk-size 2048 \
  --gpu-memory-utilization 0.9
```

在另一个终端中运行在线 benchmark：

```bash
python online_bench.py \
  --model "$MODEL" \
  --mode http \
  --base-url http://127.0.0.1:8000 \
  --scenario realistic-decode \
  --output result.json
```

在线并行服务需要在一个终端中按目标 DP/TP 配置启动 server。在外部 HTTP benchmark 模式下，实际 DP/TP 执行由 server 命令控制。

终端 1，启动 `DP=2, TP=2` server：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m babyvllm.entrypoints.cli \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-sequences 64 \
  --max-prefill-tokens-per-step 8192 \
  --max-prefill-chunk-size 2048 \
  --gpu-memory-utilization 0.9
```

终端 2，运行同配置在线 benchmark：

```bash
python online_bench.py \
  --model "$MODEL" \
  --mode http \
  --base-url http://127.0.0.1:8000 \
  --stream \
  --scenario realistic-decode \
  --num-requests 64 \
  --concurrency 8 \
  --workload mixed \
  --prompt-len-distribution bimodal \
  --long-prompt-ratio 0.6 \
  --short-input-len 512 \
  --long-input-len 1536 \
  --short-output-len 2048 \
  --long-output-len 4096 \
  --arrival-pattern poisson \
  --rate-rps 0.5 \
  --timeout 900 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-sequences 64 \
  --max-prefill-tokens-per-step 8192 \
  --max-prefill-chunk-size 2048 \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --seed 42 \
  --output result-qwen3-online-dp2-tp2.json
```
