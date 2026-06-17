from __future__ import annotations

import argparse
import gc
import random
import sys
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import torch

from babyvllm.sampling_params import SamplingParams


@dataclass
class Workload:
    prompt_token_ids: list[list[int]]
    sampling_params: list[SamplingParams]

    def clone_prompts(self) -> list[list[int]]:
        return [list(prompt) for prompt in self.prompt_token_ids]

    def clone_sampling_params(self) -> list[SamplingParams]:
        return [replace(params) for params in self.sampling_params]


@dataclass
class RunResult:
    dp_size: int
    tp_size: int
    requests: int
    output_tokens: int
    total_tokens: int
    total_time: float
    throughput: float
    per_rank: list[str]


def _parse_int_csv(value: str, flag_name: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be a comma-separated list of integers.") from exc


def parse_positive_int_csv(value: str, flag_name: str) -> list[int]:
    values = _parse_int_csv(value, flag_name)
    if not values:
        raise ValueError(f"{flag_name} must include at least one value.")
    if any(item <= 0 for item in values):
        raise ValueError(f"{flag_name} values must be positive integers.")
    return values


def parse_nonnegative_int_csv(value: str, flag_name: str) -> list[int]:
    values = _parse_int_csv(value, flag_name)
    if not values:
        raise ValueError(f"{flag_name} must include at least one value.")
    if any(item < 0 for item in values):
        raise ValueError(f"{flag_name} values must be non-negative integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"{flag_name} values must be unique.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run baby-vllm-basic-online offline DP/TP smoke validation or "
            "throughput benchmark."
        ),
        epilog=(
            "Examples:\n"
            "  DP smoke:      python offline_bench.py --model /path/to/Qwen3-0.6B --mode smoke --dp-sizes 2 --tp-size 1\n"
            "  DP+TP smoke:   python offline_bench.py --model /path/to/Qwen3-0.6B --mode smoke --dp-sizes 2 --tp-size 2\n"
            "  Benchmark:     run separate commands with --dp-sizes 1 and --dp-sizes 2 using the same seed/workload flags."
        ),
    )
    parser.add_argument("--model", required=True, help="Path to the local model directory.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "benchmark"),
        default="smoke",
        help="Validation mode. Smoke defaults to DP=2, TP=1; benchmark defaults to DP=1, TP=1.",
    )
    parser.add_argument(
        "--dp-sizes",
        default=None,
        help=(
            "Data parallel size for this process. Use one value only, for example "
            "1 or 2. Run separate commands to compare DP sizes."
        ),
    )
    parser.add_argument(
        "--tp-size",
        "--tensor-parallel-size",
        dest="tp_size",
        type=int,
        default=1,
        help="Tensor parallel size inside each DP replica. Default: 1.",
    )
    parser.add_argument(
        "--device-ids",
        default=None,
        help="Optional comma-separated CUDA device ids. Uses the first DP*TP ids.",
    )
    parser.add_argument("--num-seqs", type=int, default=8, help="Number of requests.")
    parser.add_argument("--min-input-len", type=int, default=16, help="Minimum prompt length.")
    parser.add_argument("--max-input-len", type=int, default=128, help="Maximum prompt length.")
    parser.add_argument("--min-output-len", type=int, default=8, help="Minimum generated-token limit.")
    parser.add_argument("--max-output-len", type=int, default=32, help="Maximum generated-token limit.")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Random token id range.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument(
        "--ignore-eos",
        dest="ignore_eos",
        action="store_true",
        default=True,
        help="Ignore EOS so random-token workloads generate to max_tokens.",
    )
    eos_group.add_argument(
        "--no-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        help="Respect EOS during generation.",
    )
    parser.add_argument("--warmup-size", type=int, default=1, help="Warmup request count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for workload generation.")
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager execution.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="LLMEngine max_num_batched_tokens.",
    )
    parser.add_argument(
        "--max-num-sequences",
        type=int,
        default=256,
        help="LLMEngine max_num_sequences.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for KV cache. Default: 0.9.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dp_sizes is None:
        args.dp_sizes = [2] if args.mode == "smoke" else [1]
    else:
        args.dp_sizes = parse_positive_int_csv(args.dp_sizes, "--dp-sizes")

    if args.device_ids is not None:
        args.device_ids = parse_nonnegative_int_csv(args.device_ids, "--device-ids")

    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive.")
    if len(args.dp_sizes) != 1:
        raise ValueError(
            "--dp-sizes accepts exactly one value per process. Run separate "
            "commands for each DP size to avoid same-process CUDA memory reuse."
        )
    if args.num_seqs <= 0:
        raise ValueError("--num-seqs must be positive.")
    if args.min_input_len <= 0 or args.max_input_len <= 0:
        raise ValueError("Input length bounds must be positive.")
    if args.min_input_len > args.max_input_len:
        raise ValueError("--min-input-len must be <= --max-input-len.")
    if args.min_output_len <= 0 or args.max_output_len <= 0:
        raise ValueError("Output length bounds must be positive.")
    if args.min_output_len > args.max_output_len:
        raise ValueError("--min-output-len must be <= --max-output-len.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")
    if args.temperature <= 1e-10:
        raise ValueError("--temperature must be greater than 1e-10.")
    if args.warmup_size < 0:
        raise ValueError("--warmup-size must be non-negative.")
    if args.max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be positive.")
    if args.max_num_sequences <= 0:
        raise ValueError("--max-num-sequences must be positive.")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")


def generate_workload(
    *,
    seed: int,
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    vocab_size: int,
    temperature: float,
    ignore_eos: bool,
) -> Workload:
    rng = random.Random(seed)
    prompt_token_ids = [
        [rng.randint(0, vocab_size - 1) for _ in range(rng.randint(min_input_len, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=ignore_eos,
            max_tokens=rng.randint(min_output_len, max_output_len),
        )
        for _ in range(num_seqs)
    ]
    return Workload(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)


def required_cuda_devices(dp_size: int, tp_size: int) -> int:
    return dp_size * tp_size


def select_device_ids(
    device_ids: Sequence[int] | None,
    *,
    dp_size: int,
    tp_size: int,
) -> list[int] | None:
    if device_ids is None:
        return None
    required = required_cuda_devices(dp_size, tp_size)
    if len(device_ids) < required:
        raise ValueError(
            f"--device-ids must include at least {required} ids for "
            f"data_parallel_size={dp_size} and tensor_parallel_size={tp_size}."
        )
    return list(device_ids[:required])


def preflight_cuda_devices(
    *,
    dp_size: int,
    tp_size: int,
    device_ids: Sequence[int] | None = None,
    visible_cuda_devices: int | None = None,
) -> None:
    visible = torch.cuda.device_count() if visible_cuda_devices is None else visible_cuda_devices
    required = required_cuda_devices(dp_size, tp_size)
    if visible < required:
        raise ValueError(
            "data_parallel_size*tensor_parallel_size requires "
            f"{required} CUDA devices, but only {visible} are visible."
        )
    selected_device_ids = select_device_ids(device_ids, dp_size=dp_size, tp_size=tp_size)
    if selected_device_ids and max(selected_device_ids) >= visible:
        raise ValueError(
            f"--device-ids references cuda:{max(selected_device_ids)}, "
            f"but only {visible} CUDA devices are visible."
        )


def active_dp_ranks(num_prompts: int, dp_size: int) -> list[int]:
    return [rank for rank in range(dp_size) if any(idx % dp_size == rank for idx in range(num_prompts))]


def validate_smoke_outputs(outputs: Sequence[object], expected_count: int) -> None:
    if len(outputs) != expected_count:
        raise AssertionError(f"Expected {expected_count} outputs, got {len(outputs)}.")
    for idx, output in enumerate(outputs):
        if output is None:
            raise AssertionError(f"Output at index {idx} is missing.")


def validate_smoke_metrics(
    metrics: dict,
    *,
    dp_size: int,
    expected_active_ranks: Iterable[int],
) -> None:
    for key in ("total_tokens", "total_time", "throughput"):
        if key not in metrics:
            raise AssertionError(f"Metrics missing required key: {key}")
    if metrics["total_tokens"] <= 0:
        raise AssertionError("Metrics total_tokens must be positive.")
    if metrics["total_time"] < 0:
        raise AssertionError("Metrics total_time must be non-negative.")
    if metrics["throughput"] < 0:
        raise AssertionError("Metrics throughput must be non-negative.")

    if dp_size <= 1:
        return

    per_rank = metrics.get("per_rank")
    if not isinstance(per_rank, dict):
        raise AssertionError("DP metrics must include a per_rank dictionary.")

    rank_keys = set(per_rank.keys()) | {str(key) for key in per_rank.keys()}
    missing = [
        rank
        for rank in expected_active_ranks
        if rank not in rank_keys and str(rank) not in rank_keys
    ]
    if missing:
        raise AssertionError(f"DP metrics missing active ranks: {missing}")


def count_output_tokens(outputs: Sequence[object]) -> int:
    total = 0
    for output in outputs:
        if isinstance(output, dict) and isinstance(output.get("token_ids"), list):
            total += len(output["token_ids"])
    return total


def summarize_per_rank(metrics: dict) -> list[str]:
    per_rank = metrics.get("per_rank")
    if not isinstance(per_rank, dict) or not per_rank:
        return []

    entries = []
    for rank, rank_metrics in sorted(per_rank.items(), key=lambda item: int(item[0])):
        if not isinstance(rank_metrics, dict):
            entries.append(f"Rank {rank}: unavailable")
            continue
        tokens = rank_metrics.get("total_tokens", "?")
        throughput = rank_metrics.get("throughput")
        if isinstance(throughput, (int, float)):
            entries.append(f"Rank {rank}: {tokens} tokens, {throughput:.2f} tokens/s")
        else:
            entries.append(f"Rank {rank}: {tokens} tokens")
    return entries


def make_result(
    *,
    dp_size: int,
    tp_size: int,
    request_count: int,
    outputs: Sequence[object],
    metrics: dict,
) -> RunResult:
    return RunResult(
        dp_size=dp_size,
        tp_size=tp_size,
        requests=request_count,
        output_tokens=count_output_tokens(outputs),
        total_tokens=int(metrics["total_tokens"]),
        total_time=float(metrics["total_time"]),
        throughput=float(metrics["throughput"]),
        per_rank=summarize_per_rank(metrics),
    )


def format_results_table(results: Sequence[RunResult]) -> str:
    blocks = []
    for result in results:
        block = [
            "baby-vllm-basic-online offline DP/TP benchmark",
            "",
            f"{'Data Parallel Size':<27}: {result.dp_size}",
            f"{'Tensor Parallel Size':<27}: {result.tp_size}",
            f"{'Total Requests':<27}: {result.requests} sequences",
            f"{'Total Output Tokens':<27}: {result.output_tokens} tokens",
            f"{'Total Tokens':<27}: {result.total_tokens} tokens",
            f"{'JCT (Job Completion Time)':<27}: {result.total_time:.2f} s",
            f"{'System Throughput':<27}: {result.throughput:.2f} tokens/s",
        ]
        if result.per_rank:
            block.extend(["", "--- Per-rank ---", *result.per_rank])
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def run_single_dp_size(args: argparse.Namespace, workload: Workload, dp_size: int) -> RunResult:
    from babyvllm.engine.llm_engine import LLMEngine

    selected_device_ids = select_device_ids(args.device_ids, dp_size=dp_size, tp_size=args.tp_size)
    preflight_cuda_devices(
        dp_size=dp_size,
        tp_size=args.tp_size,
        device_ids=selected_device_ids,
    )

    llm = None
    try:
        print(
            "\n"
            f"Initializing offline DP={dp_size}, TP={args.tp_size}, "
            f"requests={len(workload.prompt_token_ids)}"
        )
        llm = LLMEngine(
            model=args.model,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=args.tp_size,
            data_parallel_size=dp_size,
            data_parallel_device_ids=selected_device_ids,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_sequences=max(args.max_num_sequences, len(workload.prompt_token_ids)),
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        if args.warmup_size:
            warmup_size = min(args.warmup_size, len(workload.prompt_token_ids))
            warmup_prompts = [list(prompt) for prompt in workload.prompt_token_ids[:warmup_size]]
            warmup_params = [
                replace(params, max_tokens=min(params.max_tokens, 8))
                for params in workload.sampling_params[:warmup_size]
            ]
            print(f"Warmup: {warmup_size} request(s)")
            llm.generate(warmup_prompts, warmup_params)

        outputs, metrics = llm.generate(
            workload.clone_prompts(),
            workload.clone_sampling_params(),
        )
        validate_smoke_outputs(outputs, len(workload.prompt_token_ids))
        validate_smoke_metrics(
            metrics,
            dp_size=dp_size,
            expected_active_ranks=active_dp_ranks(len(workload.prompt_token_ids), dp_size),
        )
        return make_result(
            dp_size=dp_size,
            tp_size=args.tp_size,
            request_count=len(workload.prompt_token_ids),
            outputs=outputs,
            metrics=metrics,
        )
    finally:
        if llm is not None:
            llm.exit()
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_benchmark(args: argparse.Namespace) -> list[RunResult]:
    workload = generate_workload(
        seed=args.seed,
        num_seqs=args.num_seqs,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
        vocab_size=args.vocab_size,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
    )
    return [run_single_dp_size(args, workload, args.dp_sizes[0])]


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        print(f"baby-vllm-basic-online offline DP/TP {args.mode}")
        print(
            f"model={args.model}, dp_sizes={args.dp_sizes}, "
            f"tp_size={args.tp_size}, seed={args.seed}"
        )
        results = run_benchmark(args)
        print("\nResults")
        print(format_results_table(results))
        return 0
    except (AssertionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
