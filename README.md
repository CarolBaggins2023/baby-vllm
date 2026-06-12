# baby-vllm

Minimal offline inference engine experiments.

## Offline Data Parallel Example

`data_parallel_size` creates full model replicas. Each replica owns its own
scheduler and KV cache, and each request is routed to exactly one replica. It
can be combined with tensor parallelism; by default the engine uses
`data_parallel_size * tensor_parallel_size` CUDA devices.

```python
from babyvllm import LLMEngine, SamplingParams

llm = LLMEngine(
    model="/path/to/Qwen3-0.6B",
    data_parallel_size=2,
    tensor_parallel_size=1,
    max_num_batched_tokens=4096,
)

outputs, metrics = llm.generate(
    ["Hello", "The capital of France is", "The future of AI is"],
    SamplingParams(max_tokens=32),
)

print([output["text"] for output in outputs])
print(metrics["throughput"], metrics["per_rank"])
llm.exit()
```

## Offline Data Parallel Validation

Use the DP validation in layers:

```powershell
python -m pytest -q tests/test_data_parallel.py tests/test_dp_benchmark.py
```

Run a small real-GPU smoke test after the CPU tests pass. This checks that DP
workers initialize, generate, return per-rank metrics, and clean up.

```powershell
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode smoke --dp-sizes 2 --num-seqs 8 --max-output-len 8
```

Run larger throughput benchmarks as separate Python processes. Use the same
seed and workload flags for each DP size, then compare the printed
`Throughput(tok/s)` values manually. This avoids same-process CUDA memory reuse
between different DP configurations.

```powershell
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode benchmark --dp-sizes 1 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42

python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode benchmark --dp-sizes 2 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42
```

For DP combined with TP, the run needs `data_parallel_size * tensor_parallel_size`
visible CUDA devices. Use `--device-ids 0,1,2,3` to pin physical device order;
each run uses the first `DP * TP` ids from that list.
