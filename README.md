# baby-vllm

English | [简体中文](README.zh-CN.md)

baby-vllm is a lightweight, learning-oriented LLM inference project based on nano-vllm. This repository is intentionally small and educational, focusing on making the core mechanics of paged KV cache management, scheduling, chunked prefill, and online request handling readable. The goal is to make several important serving ideas easier to study, modify, and reason about in a compact codebase.

## Highlights

- Built from the simple nano-vllm baseline.
- Adds scheduler-level continuous batching: prefill and decode can be scheduled together, but are still executed as separate model-runner calls.
- Adds chunked prefill to avoid letting long prompts monopolize a whole engine step.
- Adds online HTTP serving for basic streaming inference experiments.
- Adds single-node tensor parallelism (TP) and data parallelism (DP) for local multi-GPU offline and online inference.

## Branches

The branches in this repository also record the order in which the project was implemented and studied.

| Branch | Description |
| --- | --- |
| `basic` | The baseline branch, mostly equivalent to nano-vllm. |
| `feature-ChunkedPrefill` | Adds continuous batching and chunked prefill on top of `basic`. |
| `feature-BatchReorder` | Experiments with batch reordering on top of `feature-ChunkedPrefill`; this idea was later abandoned in the final version. |
| `basic-online` | Adds online inference on top of `basic`. |
| `basic-dp` | Adds DP and TP on top of `basic`. |
| `basic-online-dp` | Adds DP and TP on top of `basic-online`. |
| `feature-Online` | Adds online inference on top of `feature-ChunkedPrefill`. |
| `feature-online-dp` | Adds DP and TP on top of `feature-Online`. |
| `main` | Currently the same line of work as `feature-online-dp`. |

## Model Support

baby-vllm currently supports Qwen3 models only. Users can add support for other model families in the same way as nano-vllm, by adding the corresponding model implementation and wiring it into the model runner. The benchmark results below were collected with Qwen3 0.6B, Qwen3 4B, and Qwen3 8B.

## Online Benchmark Results

Hardware: NVIDIA RTX 4090D.

Scenario: HTTP streaming mode, 64 requests, Poisson arrival, 8 concurrent clients, mixed short/long prompts, `max_model_len=8192`, `max_num_batched_tokens=16384`, `max_num_sequences=64`, `max_prefill_tokens_per_step=8192`, and `max_prefill_chunk_size=2048`.

| Model | Total Tokens | Wall Time (s) | Throughput (tok/s) | RPS | Avg TTFT (ms) | P90 TTFT (ms) | Avg TPOT (ms/token) | P90 TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B | 206,099 | 133.65 | 1,542.03 | 0.479 | 43.68 | 38.96 | 6.47 | 7.08 | 15.29 | 25.23 | 73.39 |
| Qwen3 4B | 206,099 | 311.86 | 660.87 | 0.205 | 92.11 | 112.20 | 15.48 | 16.13 | 36.30 | 59.45 | 87.93 |
| Qwen3 8B | 206,099 | 483.50 | 426.27 | 0.132 | 1,419.05 | 2,895.37 | 23.59 | 24.80 | 56.51 | 89.53 | 91.94 |

## Parallel Online Benchmark Results

Hardware: NVIDIA RTX 4090D.

Scenario: same as the single-GPU online benchmark above, HTTP streaming mode, 64 requests, Poisson arrival, 8 concurrent clients, mixed short/long prompts, `max_model_len=8192`, `max_num_batched_tokens=16384`, `max_num_sequences=64`, `max_prefill_tokens_per_step=8192`, and `max_prefill_chunk_size=2048`. Speedup is computed against the single-GPU result for the same model in the previous table.

| Model | DP | TP | Throughput (tok/s) | Speedup vs 1 GPU | RPS | Avg TTFT (ms) | Avg TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
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

## High-Pressure Qwen3 8B Online Benchmark Results

Hardware: NVIDIA RTX 4090D.

Scenario: Qwen3 8B, HTTP streaming mode, 128 requests, Poisson arrival at 1.0 target requests/s, 16 concurrent clients, mixed short/long prompts, `max_model_len=8192`, `max_num_batched_tokens=16384`, `max_num_sequences=128`, `max_prefill_tokens_per_step=8192`, and `max_prefill_chunk_size=2048`. Speedup is computed against the high-pressure `DP=1, TP=1` baseline.

| Model | DP | TP | Throughput (tok/s) | Speedup vs DP1TP1 | RPS | Avg TTFT (ms) | P90 TTFT (ms) | Avg TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 8B | 1 | 1 | 454.96 | 1.00x | 0.135 | 35,335.35 | 66,050.15 | 31.34 | 107.99 | 144.53 | 91.27 |
| Qwen3 8B | 2 | 1 | 764.05 | 1.68x | 0.227 | 2,499.55 | 8,195.31 | 25.31 | 63.51 | 96.78 | 82.89 |
| Qwen3 8B | 1 | 2 | 925.98 | 2.04x | 0.275 | 204.21 | 240.15 | 21.93 | 53.31 | 84.40 | 85.57 |
| Qwen3 8B | 2 | 2 | 1,065.36 | 2.34x | 0.316 | 190.27 | 243.02 | 18.64 | 45.39 | 69.57 | 73.30 |


## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Set `MODEL` to a local Qwen3 model path or a Hugging Face model identifier:

```bash
export MODEL=/path/to/Qwen3
```

Start the server in one terminal:

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

Run the online benchmark in another terminal:

```bash
python online_bench.py \
  --model "$MODEL" \
  --mode http \
  --base-url http://127.0.0.1:8000 \
  --scenario realistic-decode \
  --output result.json
```

For parallel online serving, start the server with the desired DP/TP sizes in one terminal. In external HTTP benchmark mode, the server command controls the actual DP/TP execution.

Terminal 1, start a `DP=2, TP=2` server:

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

Terminal 2, run the same-configuration online benchmark:

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
