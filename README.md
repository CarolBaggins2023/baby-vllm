# baby-vllm

baby-vllm is a lightweight, learning-oriented LLM inference project based on
nano-vllm. This repository is intentionally small and educational, focusing on
making the core mechanics of paged KV cache management, scheduling, chunked
prefill, and online request handling readable. The goal is to make several
important serving ideas easier to study, modify, and reason about in a compact
codebase.

## Highlights

- Built from the simple nano-vllm baseline.
- Adds continuous batching at the scheduler level.
- Adds chunked prefill to avoid letting long prompts monopolize a whole engine
  step.
- Adds online HTTP serving for basic streaming inference experiments.

## Design Notes

### Continuous Batching

Continuous batching lets the engine accept new requests while existing requests
are still decoding. In a typical serving workload, this avoids waiting for the
whole batch to finish before scheduling newly arrived prompts.

The implementation in baby-vllm is intentionally simpler than vLLM's
implementation. baby-vllm schedules prefill sequences and decode sequences
together at the logical scheduler level, but it still executes prefill and decode
separately at the physical kernel/model-runner level. This is because baby-vllm
uses operators from flash-attn and does not port the custom mixed prefill/decode
operators used by vLLM.

In other words, baby-vllm demonstrates the scheduling idea of continuous
batching, while keeping the execution path small enough for study.

### Chunked Prefill

Chunked prefill splits long prompt prefill work into smaller chunks. This helps
the scheduler interleave long-prefill requests with decode work from active
requests, reducing head-of-line blocking in online inference scenarios.

### Online Serving

The online path provides a small HTTP server and benchmark script for testing
streaming request handling under synthetic online workloads.

## Branches

The branches in this repository also record the order in which the project
was implemented and studied.

| Branch | Description |
| --- | --- |
| `basic` | The baseline branch, mostly equivalent to nano-vllm. |
| `feature-ChunkedPrefill` | Adds continuous batching and chunked prefill on top of `basic`. |
| `feature-BatchReorder` | Experiments with batch reordering on top of `feature-ChunkedPrefill`; this idea was later abandoned in the final version. |
| `basic-online` | Adds online inference on top of `basic`. |
| `feature-Online` | Adds online inference on top of `feature-ChunkedPrefill`. |
| `main` | Currently the same line of work as `feature-Online`. |

## Model Support

baby-vllm currently supports Qwen3 models only. The benchmark results below were
collected with Qwen3 0.6B, Qwen3 4B, and Qwen3 8B.

## Online Benchmark Results

Hardware: NVIDIA RTX 4090D.

Scenario: HTTP streaming mode, 64 requests, Poisson arrival,
8 concurrent clients, mixed short/long prompts, `max_model_len=8192`,
`max_num_batched_tokens=16384`, `max_num_sequences=64`,
`max_prefill_tokens_per_step=8192`, and `max_prefill_chunk_size=2048`.

| Model | Total Tokens | Wall Time (s) | Throughput (tok/s) | RPS | Avg TTFT (ms) | P90 TTFT (ms) | Avg TPOT (ms/token) | P90 TPOT (ms/token) | Avg Latency (s) | P90 Latency (s) | Avg GPU Util (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B | 206,099 | 133.65 | 1,542.03 | 0.479 | 43.68 | 38.96 | 6.47 | 7.08 | 15.29 | 25.23 | 73.39 |
| Qwen3 4B | 206,099 | 311.86 | 660.87 | 0.205 | 92.11 | 112.20 | 15.48 | 16.13 | 36.30 | 59.45 | 87.93 |
| Qwen3 8B | 206,099 | 483.50 | 426.27 | 0.132 | 1,419.05 | 2,895.37 | 23.59 | 24.80 | 56.51 | 89.53 | 91.94 |

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
