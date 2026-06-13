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

## DP and TP Communication

DP and TP use different communication mechanisms because they operate at different parallelism levels:

- **DP is coarse-grained task parallelism.** The coordinator sends prompt subsets to worker replicas and receives outputs and metrics after each worker runs the full model independently. This is **low-frequency control-plane traffic**, so it uses **Python multiprocessing pipes**.
- **TP is fine-grained model parallelism inside one replica.** TP ranks hold different model shards and must exchange GPU tensors during forward passes with collectives such as all-reduce or all-gather. This is **frequent, large, latency-sensitive data-plane traffic**, so each replica initializes a **`torch.distributed` NCCL process group** with its own **rendezvous URL**.

## Offline DP/TP Validation

Use `bench_data_parallel.py` as the supported real-GPU validation and
benchmark entrypoint for DP, TP, and combined DP+TP runs. Each command runs one
DP size and one TP size; use separate Python processes with the same seed and
workload flags when comparing configurations.

Smoke mode keeps the workload small and checks that workers initialize,
generate one populated output per prompt, return metrics, and clean up.

```bash
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode smoke --dp-sizes 2 --num-seqs 8 --max-output-len 8
```

Before running on GPUs, the sequence transport check can be run on CPU. This
guards the preemption path used by larger TP workloads.

```bash
python validate_sequence_transport.py
```

For a quick tensor-parallel smoke run, request `--tp-size > 1`.

```bash
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode smoke --dp-sizes 1 --tp-size 2 --num-seqs 4 --max-output-len 8
```

For a combined DP+TP smoke run, make `data_parallel_size * tensor_parallel_size`
CUDA devices visible.

```bash
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode smoke --dp-sizes 2 --tp-size 2 --device-ids 0,1,2,3 --num-seqs 8 --max-output-len 8
```

Run larger throughput benchmarks as separate Python processes. Use the same
seed and workload flags, then compare the printed `System Throughput` values
manually. This avoids same-process CUDA memory reuse between configurations.

```bash
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode benchmark --dp-sizes 1 --tp-size 1 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42

python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode benchmark --dp-sizes 2 --tp-size 1 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42
```

For TP throughput comparison, keep the DP size fixed and vary
`--tp-size` across separate commands.

```bash
python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode benchmark --dp-sizes 1 --tp-size 1 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42

python bench_data_parallel.py --model /path/to/Qwen3-0.6B --mode benchmark --dp-sizes 1 --tp-size 2 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42
```

For a larger model such as Qwen3-8B on two 24GB GPUs, validate the TP path with
`DP=1, TP=2`. A single-GPU run with the same long workload may fail because the
remaining KV cache capacity is too small after loading the 8B model weights.

```bash
export MODEL=/path/to/Qwen3-8B

CUDA_VISIBLE_DEVICES=0,1 python bench_data_parallel.py --model "$MODEL" --mode benchmark --dp-sizes 1 --tp-size 2 --num-seqs 256 --min-input-len 100 --max-input-len 1024 --min-output-len 100 --max-output-len 1024 --seed 42
```

Use `--device-ids 0,1,2,3` to pin physical device order; each run uses the
first `DP * TP` ids from that list. Layer-level TP microbenchmarks can be added
later for performance diagnosis, but the primary validation path is this
end-to-end DP/TP benchmark entrypoint.
