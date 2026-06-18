"""
Online inference benchmark script for baby-vllm (chunked prefill + continuous batching).

Tests the async/online inference path with two modes:
  - direct:  Uses AsyncLLMEngine directly (no HTTP overhead)
  - http:    Sends HTTP requests to a running API server

Supports burst, stagger, continuous, batch, and poisson arrival patterns.
Collects per-request TTFT, TPOT, total latency, and aggregate throughput metrics.

Usage:
    # 直接引擎模式（默认）
    python online_bench.py --model /path/to/model
    # 带更高并发
    python online_bench.py --model /path/to/model --num-requests 512 --concurrency 64
    # HTTP 模式（使用外部服务器）
    python online_bench.py --model /path/to/model --mode http --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from random import randint, seed as py_seed
from typing import Optional

import numpy as np
import torch


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class PerRequestMetrics:
    """Timing and token metrics for a single request."""
    request_id: int
    status: str = "success"  # "success" | "error" | "timeout"
    submit_time: float = 0.0
    first_token_time: float = 0.0
    completion_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft: float = 0.0
    tpot: float = 0.0
    total_time: float = 0.0
    error_message: str = ""
    request_type: str = "unknown"


@dataclass
class AggregateMetrics:
    """Aggregated benchmark statistics across all requests."""
    num_requests: int = 0
    num_success: int = 0
    num_failed: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_wall_time: float = 0.0
    throughput: float = 0.0
    requests_per_second: float = 0.0
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p99: float = 0.0
    tpot_avg: float = 0.0
    tpot_p50: float = 0.0
    tpot_p90: float = 0.0
    tpot_p99: float = 0.0
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p99: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    avg_gpu_utilization: float = 0.0


SCENARIO_PRESETS = {
    "realistic-decode": {
        "num_requests": 64,
        "concurrency": 8,
        "workload": "mixed",
        "prompt_len_distribution": "bimodal",
        "long_prompt_ratio": 0.6,
        "short_input_len": 512,
        "long_input_len": 1536,
        "short_output_len": 2048,
        "long_output_len": 4096,
        "arrival_pattern": "poisson",
        "rate_rps": 0.5,
        "timeout": 900.0,
        "max_model_len": 8192,
        "max_num_batched_tokens": 16384,
        "max_num_sequences": 64,
        "max_prefill_tokens_per_step": 8192,
        "max_prefill_chunk_size": 2048,
    },
}


def apply_scenario_preset(
    args: argparse.Namespace,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """Apply a scenario preset without clobbering explicit CLI overrides."""

    argv = sys.argv[1:] if argv is None else argv
    preset = SCENARIO_PRESETS.get(args.scenario, {})
    provided_options = set(arg.split("=", 1)[0] for arg in argv if arg.startswith("--"))

    for attr, value in preset.items():
        option = "--" + attr.replace("_", "-")
        if option not in provided_options:
            setattr(args, attr, value)

    return args


# ===========================================================================
# Test Data Generation (mirrors offline_bench.py pattern)
# ===========================================================================

def generate_random_test_data(
    num_requests: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    vocab_size: int,
    seed_val: int = 42,
) -> tuple:
    """
    Generate random prompt token IDs and SamplingParams for benchmarking.

    Mirrors the data generation pattern in offline_bench.py:
      - Fixed random seed for reproducibility
      - Random prompt token IDs with varying lengths
      - Random SamplingParams with varying max_tokens, ignore_eos=True

    Returns:
        (prompt_token_ids_list, sampling_params_list): tuple of two lists.
    """
    from babyvllm import SamplingParams

    py_seed(seed_val)

    prompt_token_ids = [
        [
            randint(0, vocab_size - 1)
            for _ in range(randint(min_input_len, max_input_len))
        ]
        for _ in range(num_requests)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(min_output_len, max_output_len),
        )
        for _ in range(num_requests)
    ]
    return prompt_token_ids, sampling_params, ["random"] * num_requests


def _parse_int_csv(value: str, flag_name: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be a comma-separated list of integers.") from exc


def parse_nonnegative_int_csv(value: str, flag_name: str) -> list[int]:
    values = _parse_int_csv(value, flag_name)
    if not values:
        raise ValueError(f"{flag_name} must include at least one value.")
    if any(item < 0 for item in values):
        raise ValueError(f"{flag_name} values must be non-negative integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"{flag_name} values must be unique.")
    return values


def required_cuda_devices(dp_size: int, tp_size: int) -> int:
    return dp_size * tp_size


def select_device_ids(
    device_ids: Optional[list[int]],
    *,
    dp_size: int,
    tp_size: int,
) -> Optional[list[int]]:
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
    device_ids: Optional[list[int]] = None,
) -> None:
    visible = torch.cuda.device_count()
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


def generate_mixed_test_data(
    num_requests: int,
    prompt_len_distribution: str,
    short_input_len: int,
    long_input_len: int,
    short_output_len: int,
    long_output_len: int,
    long_prompt_ratio: float,
    seed_val: int = 42,
) -> tuple:
    """Generate a realistic long-prompt-heavy online workload."""
    from babyvllm import SamplingParams

    rng = random.Random(seed_val)
    short_input_len = max(short_input_len, 1)
    long_input_len = max(long_input_len, short_input_len)
    short_output_len = max(short_output_len, 1)
    long_output_len = max(long_output_len, short_output_len)
    long_prompt_ratio = min(max(long_prompt_ratio, 0.0), 1.0)

    def sample_prompt_len() -> tuple[int, str]:
        if prompt_len_distribution == "bimodal":
            is_long = rng.random() < long_prompt_ratio
            if is_long:
                low = max(short_input_len + 1, int(long_input_len * 0.75))
                return rng.randint(low, long_input_len), "long"
            low = max(1, int(short_input_len * 0.5))
            return rng.randint(low, short_input_len), "short"
        if prompt_len_distribution == "uniform":
            length = rng.randint(short_input_len, long_input_len)
            split = short_input_len + (long_input_len-short_input_len) * 0.5
            return length, "long" if length >= split else "short"
        if prompt_len_distribution == "lognormal":
            mu = (np.log(short_input_len) + np.log(long_input_len)) / 2
            sigma = 1.0
            length = int(rng.lognormvariate(mu, sigma))
            length = max(short_input_len, min(length, long_input_len))
            split = short_input_len + (long_input_len-short_input_len) * 0.5
            return length, "long" if length >= split else "short"
        raise ValueError(f"Unknown prompt length distribution: {prompt_len_distribution}")

    prompt_token_ids = []
    sampling_params = []
    request_types = []

    for _ in range(num_requests):
        prompt_len, request_type = sample_prompt_len()
        if request_type == "long":
            max_tokens = rng.randint(max(1, long_output_len // 2), long_output_len)
        else:
            max_tokens = rng.randint(max(1, short_output_len // 2), short_output_len)
        prompt_token_ids.append([rng.randint(0, 10000) for _ in range(prompt_len)])
        sampling_params.append(SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=max_tokens,
        ))
        request_types.append(request_type)

    return prompt_token_ids, sampling_params, request_types


# ===========================================================================
# Metrics Computation
# ===========================================================================

def compute_aggregate_metrics(
    per_request_list: list,
    wall_start: float,
    wall_end: float,
    avg_gpu_memory_mb: float = 0.0,
    avg_gpu_utilization: float = 0.0,
) -> AggregateMetrics:
    """
    Compute aggregate statistics from per-request metrics.

    For TTFT, TPOT, and latency percentiles, only successful requests are included.
    Failed/timeout requests contribute to counts and tokens but not timing distributions.
    """
    success_list = [r for r in per_request_list if r.status == "success"]
    agg = AggregateMetrics()
    agg.num_requests = len(per_request_list)
    agg.num_success = len(success_list)
    agg.num_failed = agg.num_requests - agg.num_success

    agg.total_prompt_tokens = sum(r.prompt_tokens for r in per_request_list)
    agg.total_completion_tokens = sum(r.completion_tokens for r in per_request_list)
    agg.total_tokens = agg.total_prompt_tokens + agg.total_completion_tokens

    agg.total_wall_time = wall_end - wall_start
    if agg.total_wall_time > 0:
        agg.throughput = agg.total_tokens / agg.total_wall_time
        agg.requests_per_second = agg.num_success / agg.total_wall_time

    if success_list:
        ttft_vals = np.array([r.ttft for r in success_list])
        tpot_vals = np.array([r.tpot for r in success_list])
        latency_vals = np.array([r.total_time for r in success_list])

        agg.ttft_avg = float(np.mean(ttft_vals))
        agg.ttft_p50 = float(np.percentile(ttft_vals, 50))
        agg.ttft_p90 = float(np.percentile(ttft_vals, 90))
        agg.ttft_p99 = float(np.percentile(ttft_vals, 99))

        agg.tpot_avg = float(np.mean(tpot_vals))
        agg.tpot_p50 = float(np.percentile(tpot_vals, 50))
        agg.tpot_p90 = float(np.percentile(tpot_vals, 90))
        agg.tpot_p99 = float(np.percentile(tpot_vals, 99))

        agg.latency_avg = float(np.mean(latency_vals))
        agg.latency_p50 = float(np.percentile(latency_vals, 50))
        agg.latency_p90 = float(np.percentile(latency_vals, 90))
        agg.latency_p99 = float(np.percentile(latency_vals, 99))

    agg.avg_gpu_memory_mb = avg_gpu_memory_mb
    agg.avg_gpu_utilization = avg_gpu_utilization

    return agg


def compute_request_type_breakdown(per_request_list: list) -> dict:
    """Compute latency/token summaries by request type (short/long/random)."""
    breakdown = {}
    request_types = sorted({r.request_type for r in per_request_list})
    for request_type in request_types:
        group = [r for r in per_request_list if r.request_type == request_type]
        successful = [r for r in group if r.status == "success"]
        ttfts = [r.ttft for r in successful if r.ttft > 0]
        tpots = [r.tpot for r in successful if r.tpot > 0]
        latencies = [r.total_time for r in successful if r.total_time > 0]
        prompt_tokens = [r.prompt_tokens for r in successful]
        completion_tokens = [r.completion_tokens for r in successful]

        def pct(values: list[float], q: int) -> float:
            return float(np.percentile(values, q)) if values else 0.0

        breakdown[request_type] = {
            "num_requests": len(group),
            "num_success": len(successful),
            "num_failed": len(group) - len(successful),
            "prompt_tokens_avg": float(np.mean(prompt_tokens)) if prompt_tokens else 0.0,
            "completion_tokens_avg": float(np.mean(completion_tokens)) if completion_tokens else 0.0,
            "ttft": {
                "avg": float(np.mean(ttfts)) if ttfts else 0.0,
                "p50": pct(ttfts, 50),
                "p90": pct(ttfts, 90),
                "p99": pct(ttfts, 99),
            },
            "tpot": {
                "avg": float(np.mean(tpots)) if tpots else 0.0,
                "p50": pct(tpots, 50),
                "p90": pct(tpots, 90),
                "p99": pct(tpots, 99),
            },
            "latency": {
                "avg": float(np.mean(latencies)) if latencies else 0.0,
                "p50": pct(latencies, 50),
                "p90": pct(latencies, 90),
                "p99": pct(latencies, 99),
            },
        }
    return breakdown


# ===========================================================================
# Console Output
# ===========================================================================

def print_report(config: argparse.Namespace, agg: AggregateMetrics) -> None:
    """Format and print benchmark results, matching offline_bench.py console style."""
    mode_str = config.mode
    if config.mode == "http" and config.server_embedded:
        mode_str = "http (embedded)"
    elif config.mode == "http":
        mode_str = "http (external)"

    print("\n" + "=" * 66)
    print(" baby-vllm Online Benchmark Results")
    print("=" * 66)
    print(f" Mode                  : {mode_str}")
    print(f" Total Requests        : {agg.num_requests} "
          f"({agg.num_success} success, {agg.num_failed} failed)")
    print(f" Concurrency           : {config.concurrency}")
    print(f" Streaming             : {config.stream}")
    print(f" Arrival Pattern       : {config.arrival_pattern}")
    if config.arrival_pattern == "batch":
        print(f" Batch Size / Interval : {config.batch_size} / "
              f"{config.batch_interval:.1f}s")
    elif config.arrival_pattern == "poisson" or config.arrival_pattern == "continuous":
        print(f" Target Rate           : {config.rate_rps} req/s")

    print("-" * 66)
    print(f" Prompt Tokens         : {agg.total_prompt_tokens:,}")
    print(f" Completion Tokens     : {agg.total_completion_tokens:,}")
    print(f" Total Tokens          : {agg.total_tokens:,}")
    print(f" Wall Time             : {agg.total_wall_time:.2f} s")
    print(f" Throughput            : {agg.throughput:.1f} tokens/s")
    print(f" Request Rate          : {agg.requests_per_second:.1f} req/s")
    if agg.avg_gpu_memory_mb > 0:
        print(f" Avg GPU Memory        : {agg.avg_gpu_memory_mb:.1f} MB")
    if agg.avg_gpu_utilization > 0:
        print(f" Avg GPU Utilization   : {agg.avg_gpu_utilization:.1f} %")

    print("-" * 66)
    print("--- TTFT (Time To First Token) ---")
    print(f" Avg   : {agg.ttft_avg:.4f} s")
    print(f" P50   : {agg.ttft_p50:.4f} s")
    print(f" P90   : {agg.ttft_p90:.4f} s")
    print(f" P99   : {agg.ttft_p99:.4f} s")

    print("-" * 66)
    print("--- TPOT (Time Per Output Token) ---")
    print(f" Avg   : {agg.tpot_avg:.4f} s")
    print(f" P50   : {agg.tpot_p50:.4f} s")
    print(f" P90   : {agg.tpot_p90:.4f} s")
    print(f" P99   : {agg.tpot_p99:.4f} s")

    print("-" * 66)
    print("--- Per-Request Latency ---")
    print(f" Avg   : {agg.latency_avg:.4f} s")
    print(f" P50   : {agg.latency_p50:.4f} s")
    print(f" P90   : {agg.latency_p90:.4f} s")
    print(f" P99   : {agg.latency_p99:.4f} s")
    print("=" * 66)


def print_engine_stats(engine_stats: dict) -> None:
    """Print lightweight engine instrumentation counters if available."""
    if not engine_stats:
        return

    scheduler_stats = engine_stats.get("scheduler", {})
    model_runner_stats = engine_stats.get("model_runner", {})

    print("\n--- Engine Stats ---")
    if scheduler_stats:
        print(
            " Scheduler batches: "
            f"pure_decode={scheduler_stats.get('pure_decode', 0)} | "
            f"pure_prefill={scheduler_stats.get('pure_prefill', 0)} | "
            f"mixed={scheduler_stats.get('mixed', 0)} | "
            f"preempt={scheduler_stats.get('preempt', 0)}"
        )
    if model_runner_stats:
        print(
            " Model runner: "
            f"cuda_graph_replay={model_runner_stats.get('cuda_graph_replay', 0)} | "
            f"eager={model_runner_stats.get('eager', 0)}"
        )


def print_request_type_breakdown(per_request_list: list) -> None:
    breakdown = compute_request_type_breakdown(per_request_list)
    if len(breakdown) <= 1:
        return
    print("\n--- By Request Type ---")
    for request_type, stats in breakdown.items():
        print(
            f" {request_type:>6}: n={stats['num_success']:3d}/{stats['num_requests']:3d} | "
            f"prompt_avg={stats['prompt_tokens_avg']:.1f} | "
            f"ttft_p50={stats['ttft']['p50']:.4f}s | "
            f"tpot_p50={stats['tpot']['p50']:.4f}s | "
            f"lat_p90={stats['latency']['p90']:.4f}s"
        )


# ===========================================================================
# JSON Export
# ===========================================================================

def export_json(
    agg: AggregateMetrics,
    per_request_list: list,
    filepath: str,
    config: Optional[argparse.Namespace] = None,
    engine_stats: Optional[dict] = None,
) -> None:
    """Export benchmark results to JSON file."""
    result = {
        "config": {
            "mode": config.mode if config else "unknown",
            "scenario": getattr(config, "scenario", "realistic-decode"),
            "num_requests": agg.num_requests,
            "concurrency": getattr(config, "concurrency", 0),
            "stream": getattr(config, "stream", True),
            "arrival_pattern": getattr(config, "arrival_pattern", "poisson"),
            "rate_rps": getattr(config, "rate_rps", 0.0),
            "batch_size": getattr(config, "batch_size", 32),
            "batch_interval": getattr(config, "batch_interval", 5.0),
            "workload": getattr(config, "workload", "mixed"),
            "min_input_len": getattr(config, "min_input_len", 16),
            "max_input_len": getattr(config, "max_input_len", 1024),
            "min_output_len": getattr(config, "min_output_len", 8),
            "max_output_len": getattr(config, "max_output_len", 1024),
            "vocab_size": getattr(config, "vocab_size", 10000),
            "prompt_len_distribution": getattr(config, "prompt_len_distribution", "uniform"),
            "long_prompt_ratio": getattr(config, "long_prompt_ratio", 0.0),
            "short_input_len": getattr(config, "short_input_len", 0),
            "long_input_len": getattr(config, "long_input_len", 0),
            "short_output_len": getattr(config, "short_output_len", 0),
            "long_output_len": getattr(config, "long_output_len", 0),
            "max_model_len": getattr(config, "max_model_len", None),
            "max_num_batched_tokens": getattr(config, "max_num_batched_tokens", 0),
            "max_num_sequences": getattr(config, "max_num_sequences", 0),
            "max_prefill_tokens_per_step": getattr(config, "max_prefill_tokens_per_step", 0),
            "max_prefill_chunk_size": getattr(config, "max_prefill_chunk_size", 0),
            "data_parallel_size": getattr(config, "data_parallel_size", 1),
            "tensor_parallel_size": getattr(config, "tensor_parallel_size", 1),
            "device_ids": getattr(config, "device_ids", None),
        },
        "aggregate": {
            "num_requests": agg.num_requests,
            "num_success": agg.num_success,
            "num_failed": agg.num_failed,
            "total_prompt_tokens": agg.total_prompt_tokens,
            "total_completion_tokens": agg.total_completion_tokens,
            "total_tokens": agg.total_tokens,
            "total_wall_time": agg.total_wall_time,
            "throughput_tokens_per_sec": agg.throughput,
            "requests_per_second": agg.requests_per_second,
            "ttft": {
                "avg": agg.ttft_avg,
                "p50": agg.ttft_p50,
                "p90": agg.ttft_p90,
                "p99": agg.ttft_p99,
            },
            "tpot": {
                "avg": agg.tpot_avg,
                "p50": agg.tpot_p50,
                "p90": agg.tpot_p90,
                "p99": agg.tpot_p99,
            },
            "latency": {
                "avg": agg.latency_avg,
                "p50": agg.latency_p50,
                "p90": agg.latency_p90,
                "p99": agg.latency_p99,
            },
            "avg_gpu_memory_mb": agg.avg_gpu_memory_mb,
            "avg_gpu_utilization": agg.avg_gpu_utilization,
        },
        "by_request_type": compute_request_type_breakdown(per_request_list),
        "engine_stats": engine_stats or {},
        "per_request": [
            {
                "request_id": r.request_id,
                "request_type": r.request_type,
                "status": r.status,
                "submit_time": r.submit_time,
                "first_token_time": r.first_token_time,
                "completion_time": r.completion_time,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "ttft": r.ttft,
                "tpot": r.tpot,
                "total_time": r.total_time,
                "error_message": r.error_message,
            }
            for r in per_request_list
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults exported to: {filepath}")


# ===========================================================================
# GPU Usage Sampler
# ===========================================================================

class GPUUsageSampler:
    """
    Background asyncio task that samples GPU memory and utilization
    at regular intervals during the benchmark window.

    Uses torch.cuda.memory_stats() for memory and pynvml for utilization
    (falls back gracefully if pynvml is not available).
    """

    def __init__(self, interval_sec: float = 0.1):
        self._samples: list = []  # list of (memory_mb, utilization_pct)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._interval = interval_sec
        self._nvml_handle = None

    async def start(self) -> None:
        """Begin background GPU sampling."""
        self._running = True
        self._samples = []
        # Try to initialize pynvml for GPU utilization
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml_handle = None
        self._task = asyncio.ensure_future(self._sample_loop())

    async def _sample_loop(self) -> None:
        """Continuously sample GPU stats while running."""
        while self._running:
            try:
                # Memory from torch.cuda
                mem_stats = torch.cuda.memory_stats()
                mem_allocated = mem_stats.get(
                    "allocated_bytes.all.current", 0
                ) / (1024 * 1024)

                # Utilization from pynvml (if available)
                util_pct = 0.0
                if self._nvml_handle is not None:
                    import pynvml
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(
                        self._nvml_handle
                    )
                    util_pct = float(util_info.gpu)

                self._samples.append((mem_allocated, util_pct))
            except Exception:
                # Sampling is best-effort; suppress all errors to keep loop alive
                pass
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        """Stop GPU sampling and return final averages."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def avg_memory_mb(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[0] for s in self._samples) / len(self._samples)

    @property
    def avg_utilization(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[1] for s in self._samples) / len(self._samples)


# ===========================================================================
# Arrival pattern helper — launch tasks in batches at intervals
# ===========================================================================

async def _launch_with_arrival(
    runner,
    num_requests: int,
    prompt_token_ids_list: list,
    sampling_params_list: list,
    request_type_list: list,
    arrival_pattern: str,
    stagger_interval_sec: float,
    rate_rps: float,
    batch_size: int,
    batch_interval: float,
    seed: int = 42,
) -> list:
    """
    Create and gather tasks according to the arrival pattern.

    Five patterns:
      burst     — all requests launched at once, gated by semaphore
      stagger   — one request every stagger_interval_sec, gated by semaphore
      continuous— requests at a fixed target rate (rate_rps), gated by semaphore
      batch     — batch_size requests every batch_interval seconds; batches
                  overlap (new batch submitted without waiting for previous)
      poisson   — requests at random intervals (exponential distribution)
                  with average rate rate_rps; best for real-world simulation
    """
    tasks = []

    if arrival_pattern == "burst":
        for i in range(num_requests):
            task = asyncio.create_task(runner._run_single_request(
                i, prompt_token_ids_list[i], sampling_params_list[i], request_type_list[i]
            ))
            tasks.append(task)
        return await asyncio.gather(*tasks, return_exceptions=True)

    elif arrival_pattern == "stagger":
        for i in range(num_requests):
            task = asyncio.create_task(runner._run_single_request(
                i, prompt_token_ids_list[i], sampling_params_list[i], request_type_list[i]
            ))
            tasks.append(task)
            if i < num_requests - 1:
                await asyncio.sleep(stagger_interval_sec)
        return await asyncio.gather(*tasks, return_exceptions=True)

    elif arrival_pattern == "continuous":
        interval = 1.0 / rate_rps if rate_rps > 0 else 0.0
        for i in range(num_requests):
            task = asyncio.create_task(runner._run_single_request(
                i, prompt_token_ids_list[i], sampling_params_list[i], request_type_list[i]
            ))
            tasks.append(task)
            if i < num_requests - 1 and interval > 0:
                await asyncio.sleep(interval)
        return await asyncio.gather(*tasks, return_exceptions=True)

    elif arrival_pattern == "batch":
        # Submit batch_size requests every batch_interval seconds.
        # Requests from different batches overlap — earlier batches may still
        # be running when later batches arrive, simulating real online traffic
        # where users arrive in waves independently of request completion.
        sent = 0
        batch_num = 0
        while sent < num_requests:
            batch_end = min(sent + batch_size, num_requests)
            batch_num += 1
            if batch_num > 1:
                print(f"  Submitting batch {batch_num} "
                      f"({batch_end - sent} requests)...")
                await asyncio.sleep(batch_interval)

            for i in range(sent, batch_end):
                tasks.append(asyncio.create_task(
                    runner._run_single_request(
                        i, prompt_token_ids_list[i], sampling_params_list[i], request_type_list[i]
                    )
                ))
            sent = batch_end
        return await asyncio.gather(*tasks, return_exceptions=True)

    elif arrival_pattern == "poisson":
        # Poisson process: inter-arrival times follow an exponential distribution
        # with rate = rate_rps (mean interval = 1/rate_rps seconds).
        # Uses an independent random.Random(seed) instance so the arrival
        # sequence is fully determined by --seed and NOT affected by how many
        # random draws test-data generation consumed.  Same seed = same intervals.
        rng = random.Random(seed)
        for i in range(num_requests):
            tasks.append(asyncio.create_task(runner._run_single_request(
                i, prompt_token_ids_list[i], sampling_params_list[i], request_type_list[i]
            )))
            if i < num_requests - 1:
                interval = rng.expovariate(rate_rps) if rate_rps > 0 else 0.0
                await asyncio.sleep(interval)
        return await asyncio.gather(*tasks, return_exceptions=True)

    else:
        raise ValueError(f"Unknown arrival pattern: {arrival_pattern}")


# ===========================================================================
# Mode A: Direct Engine Runner
# ===========================================================================

class DirectEngineRunner:
    """
    Benchmark runner that uses AsyncLLMEngine directly (no HTTP).

    Each request goes through engine.generate() which is an async generator
    yielding RequestOutput objects. Timing metrics (TTFT, TPOT, total_time)
    are extracted from the final RequestOutput's fields.
    """

    def __init__(
        self,
        model: str,
        engine_kwargs: dict,
        concurrency: int,
        timeout: float,
        verbose: bool,
    ):
        self._model = model
        self._engine_kwargs = engine_kwargs
        self._concurrency = concurrency
        self._timeout = timeout
        self._verbose = verbose
        self._semaphore = asyncio.Semaphore(concurrency)
        self._engine = None

    async def _create_engine(self) -> None:
        """Lazily create the AsyncLLMEngine."""
        if self._engine is None:
            from babyvllm.engine.async_llm_engine import AsyncLLMEngine

            self._engine = AsyncLLMEngine(model=self._model, **self._engine_kwargs)

    async def _run_single_request(
        self,
        idx: int,
        prompt_token_ids: list,
        sampling_params,
        request_type: str = "unknown",
    ) -> PerRequestMetrics:
        """Execute a single request through the engine with concurrency control."""
        async with self._semaphore:
            submit_time = time.perf_counter()
            first_token_time = 0.0
            completion_time = 0.0
            final_output = None
            completion_token_ids = []

            try:
                gen = self._engine.generate(prompt_token_ids, sampling_params, request_id=idx)

                async def _consume():
                    nonlocal first_token_time, completion_time, final_output
                    async for output in gen:
                        if output.request_id != idx:
                            raise AssertionError(
                                f"Request ID mismatch: expected {idx}, got {output.request_id}."
                            )
                        if output.token_ids and first_token_time == 0.0:
                            first_token_time = time.perf_counter()
                        completion_token_ids.extend(output.token_ids)
                        if output.finished:
                            completion_time = time.perf_counter()
                            final_output = output

                await asyncio.wait_for(_consume(), timeout=self._timeout)

                if final_output is not None:
                    # Extract metrics from RequestOutput fields
                    # (populated by AsyncLLMEngine when finished=True)
                    prompt_tokens = len(final_output.prompt_token_ids)
                    completion_tokens = len(completion_token_ids)
                    ttft = (
                        final_output.ttft
                        if final_output.ttft is not None
                        else ((first_token_time - submit_time) if first_token_time else 0.0)
                    )
                    tpot = (
                        final_output.tpot
                        if final_output.tpot is not None
                        else (
                            (completion_time - first_token_time) / max(completion_tokens-1, 1)
                            if first_token_time and completion_tokens > 1 else 0.0
                        )
                    )
                    total_time = (
                        final_output.total_time
                        if final_output.total_time is not None
                        else (completion_time - submit_time)
                    )

                    if self._verbose:
                        print(
                            f"  [{idx:4d}] SUCCESS | "
                            f"prompt={prompt_tokens:5d} | "
                            f"completion={completion_tokens:5d} | "
                            f"ttft={ttft:.4f}s | "
                            f"tpot={tpot:.4f}s | "
                            f"total={total_time:.4f}s"
                        )
                    return PerRequestMetrics(
                        request_id=idx,
                        status="success",
                        submit_time=submit_time,
                        first_token_time=first_token_time,
                        completion_time=completion_time,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        ttft=ttft,
                        tpot=tpot,
                        total_time=total_time,
                        request_type=request_type,
                    )
                else:
                    if self._verbose:
                        print(f"  [{idx:4d}] ERROR   | no output generated")
                    return PerRequestMetrics(
                        request_id=idx,
                        status="error",
                        submit_time=submit_time,
                        prompt_tokens=len(prompt_token_ids),
                        error_message="No output generated from engine",
                        request_type=request_type,
                    )

            except asyncio.TimeoutError:
                if self._verbose:
                    print(f"  [{idx:4d}] TIMEOUT | {self._timeout:.1f}s limit exceeded")
                return PerRequestMetrics(
                    request_id=idx,
                    status="timeout",
                    submit_time=submit_time,
                    prompt_tokens=len(prompt_token_ids),
                    error_message=f"Timeout after {self._timeout:.1f}s",
                    request_type=request_type,
                )
            except Exception as e:
                if self._verbose:
                    print(f"  [{idx:4d}] ERROR   | {type(e).__name__}: {e}")
                return PerRequestMetrics(
                    request_id=idx,
                    status="error",
                    submit_time=submit_time,
                    prompt_tokens=len(prompt_token_ids),
                    error_message=f"{type(e).__name__}: {e}",
                    request_type=request_type,
                )

    async def get_engine_stats(self) -> dict:
        """Return instrumentation counters from the in-process engine."""
        if self._engine is None:
            return {}
        return self._engine.get_stats()

    async def run(
        self,
        prompt_token_ids_list: list,
        sampling_params_list: list,
        request_type_list: list,
        arrival_pattern: str,
        stagger_interval_sec: float,
        rate_rps: float,
        batch_size: int,
        batch_interval: float,
        seed: int = 42,
    ) -> tuple:
        """Execute all benchmark requests and return (metrics_list, wall_start, wall_end)."""
        num_requests = len(prompt_token_ids_list)

        await self._create_engine()

        # Warmup: run a single request to compile CUDA graphs and allocate memory
        from babyvllm import SamplingParams

        print("Engine warming up (compile CUDA Graph, allocate memory)...")
        async for _ in self._engine.generate(
            [1, 2, 3], SamplingParams(max_tokens=8)
        ):
            pass
        print("Warmup complete. Starting benchmark...\n")

        wall_start = time.perf_counter()

        raw_results = await _launch_with_arrival(
            self,
            num_requests=num_requests,
            prompt_token_ids_list=prompt_token_ids_list,
            sampling_params_list=sampling_params_list,
            request_type_list=request_type_list,
            arrival_pattern=arrival_pattern,
            stagger_interval_sec=stagger_interval_sec,
            rate_rps=rate_rps,
            batch_size=batch_size,
            batch_interval=batch_interval,
            seed=seed,
        )

        wall_end = time.perf_counter()

        # Convert gather results (which may contain exceptions) to PerRequestMetrics
        per_request_list = []
        for i, result in enumerate(raw_results):
            if isinstance(result, PerRequestMetrics):
                per_request_list.append(result)
            elif isinstance(result, Exception):
                per_request_list.append(PerRequestMetrics(
                    request_id=i,
                    status="error",
                    submit_time=0.0,
                    prompt_tokens=len(prompt_token_ids_list[i]),
                    error_message=f"Unhandled exception: {result}",
                    request_type=request_type_list[i],
                ))
            else:
                per_request_list.append(PerRequestMetrics(
                    request_id=i,
                    status="error",
                    submit_time=0.0,
                    prompt_tokens=len(prompt_token_ids_list[i]),
                    error_message=f"Unexpected result type: {type(result)}",
                    request_type=request_type_list[i],
                ))

        return per_request_list, wall_start, wall_end


# ===========================================================================
# Mode B: HTTP API Runner
# ===========================================================================

# Subprocess entry point for embedded server mode
def _run_embedded_server(
    model: str,
    port: int,
    engine_kwargs: dict,
) -> None:
    """Target function for embedded API server subprocess."""
    from babyvllm.engine.async_llm_engine import AsyncLLMEngine
    from babyvllm.entrypoints import api_server
    import uvicorn

    engine = AsyncLLMEngine(model=model, **engine_kwargs)
    api_server._engine = engine

    uvicorn.run(
        api_server.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


class HTTPAPIRunner:
    """
    Benchmark runner that sends HTTP requests to a baby-vllm API server.

    Supports two server modes:
      - external:  user starts the server separately (--server-embedded not set)
      - embedded:  runner spawns the server as a subprocess (--server-embedded)
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        port: int,
        server_embedded: bool,
        engine_kwargs: dict,
        stream: bool,
        concurrency: int,
        timeout: float,
        verbose: bool,
    ):
        # Pre-flight: verify httpx is available before doing anything
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "httpx is required for HTTP API mode. "
                "Install it with: pip install httpx"
            )

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._port = port
        self._server_embedded = server_embedded
        self._engine_kwargs = engine_kwargs
        self._stream = stream
        self._concurrency = concurrency
        self._timeout = timeout
        self._verbose = verbose
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = None
        self._server_process = None
        self._engine_stats = {}

    async def _wait_for_server(self, timeout: float = 120.0) -> bool:
        """Poll /health endpoint until server responds 200."""
        import httpx

        deadline = time.perf_counter() + timeout
        health_url = f"{self._base_url}/health"
        while time.perf_counter() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(health_url, timeout=1.0)
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    def _start_embedded_server(self) -> None:
        """Spawn the API server in a subprocess."""
        import multiprocessing

        self._server_process = multiprocessing.Process(
            target=_run_embedded_server,
            args=(self._model, self._port, self._engine_kwargs),
            daemon=True,
        )
        self._server_process.start()
        print(f"Embedded server starting on port {self._port}...")

    def _stop_embedded_server(self) -> None:
        """Terminate the embedded server subprocess."""
        if self._server_process is not None and self._server_process.is_alive():
            self._server_process.terminate()
            self._server_process.join(timeout=10)
            if self._server_process.is_alive():
                self._server_process.kill()
                self._server_process.join()
            print("Embedded server stopped.")

    async def get_engine_stats(self) -> dict:
        """Return instrumentation counters from the HTTP server."""
        if self._client is None:
            return dict(self._engine_stats)

        try:
            response = await self._client.get(
                f"{self._base_url}/debug/stats",
                timeout=5.0,
            )
            if response.status_code == 200:
                self._engine_stats = response.json()
        except Exception as e:
            if self._verbose:
                print(f"Failed to fetch engine stats: {type(e).__name__}: {e}")
        return dict(self._engine_stats)

    async def _run_single_request(
        self,
        idx: int,
        prompt_token_ids: list,
        sampling_params,
        request_type: str = "unknown",
    ) -> PerRequestMetrics:
        """Execute a single request via HTTP with concurrency control."""
        async with self._semaphore:
            submit_time = time.perf_counter()
            first_token_time = 0.0
            completion_time = 0.0
            prompt_tokens = len(prompt_token_ids)
            completion_tokens = 0
            ttft = 0.0
            tpot = 0.0
            total_time = 0.0

            import httpx

            body = {
                "model": "baby-vllm",
                "prompt": prompt_token_ids,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "ignore_eos": sampling_params.ignore_eos,
                "stream": self._stream,
            }

            try:
                if self._stream:
                    async with self._client.stream(
                        "POST",
                        f"{self._base_url}/v1/completions",
                        json=body,
                        timeout=self._timeout,
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            return PerRequestMetrics(
                                request_id=idx,
                                status="error",
                                submit_time=submit_time,
                                prompt_tokens=prompt_tokens,
                                error_message=(
                                    f"HTTP {response.status_code}: "
                                    f"{error_text.decode('utf-8', errors='replace')[:200]}"
                                ),
                                request_type=request_type,
                            )

                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            if line == "data: [DONE]":
                                continue
                            data_str = line[len("data: "):]
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if first_token_time == 0.0:
                                choices = chunk.get("choices") or []
                                text_delta = ""
                                if choices:
                                    text_delta = choices[0].get("text") or ""
                                if text_delta:
                                    first_token_time = time.perf_counter()

                            # Check for final chunk (has metrics)
                            if "metrics" in chunk and chunk["metrics"] is not None:
                                completion_time = time.perf_counter()
                                m = chunk["metrics"]
                                ttft = m.get("ttft") or 0.0
                                tpot = m.get("tpot") or 0.0
                                total_time = m.get("total_time") or 0.0

                            if "usage" in chunk and chunk["usage"] is not None:
                                completion_tokens = (
                                    chunk["usage"].get("completion_tokens", 0)
                                )
                else:
                    # Non-streaming: single JSON response
                    response = await self._client.post(
                        f"{self._base_url}/v1/completions",
                        json=body,
                        timeout=self._timeout,
                    )
                    if response.status_code != 200:
                        error_text = response.text
                        return PerRequestMetrics(
                            request_id=idx,
                            status="error",
                            submit_time=submit_time,
                            prompt_tokens=prompt_tokens,
                            error_message=(
                                f"HTTP {response.status_code}: "
                                f"{error_text[:200]}"
                            ),
                            request_type=request_type,
                        )

                    data = response.json()
                    first_token_time = time.perf_counter()
                    completion_time = first_token_time

                    if "usage" in data:
                        usage = data["usage"]
                        completion_tokens = usage.get("completion_tokens", 0)
                    if "metrics" in data and data["metrics"] is not None:
                        m = data["metrics"]
                        ttft = m.get("ttft") or 0.0
                        tpot = m.get("tpot") or 0.0
                        total_time = m.get("total_time") or 0.0
                    else:
                        # Fallback: compute from wall clock
                        total_time = completion_time - submit_time
                        ttft = total_time
                        tpot = total_time / max(completion_tokens, 1)

                if self._verbose:
                    print(
                        f"  [{idx:4d}] SUCCESS | "
                        f"prompt={prompt_tokens:5d} | "
                        f"completion={completion_tokens:5d} | "
                        f"ttft={ttft:.4f}s | "
                        f"tpot={tpot:.4f}s | "
                        f"total={total_time:.4f}s"
                    )
                return PerRequestMetrics(
                    request_id=idx,
                    status="success",
                    submit_time=submit_time,
                    first_token_time=first_token_time,
                    completion_time=completion_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft=ttft,
                    tpot=tpot,
                    total_time=total_time,
                    request_type=request_type,
                )

            except httpx.TimeoutException:
                if self._verbose:
                    print(f"  [{idx:4d}] TIMEOUT | {self._timeout:.1f}s limit exceeded")
                return PerRequestMetrics(
                    request_id=idx,
                    status="timeout",
                    submit_time=submit_time,
                    prompt_tokens=prompt_tokens,
                    error_message=f"Timeout after {self._timeout:.1f}s",
                    request_type=request_type,
                )
            except Exception as e:
                if self._verbose:
                    print(f"  [{idx:4d}] ERROR   | {type(e).__name__}: {e}")
                return PerRequestMetrics(
                    request_id=idx,
                    status="error",
                    submit_time=submit_time,
                    prompt_tokens=prompt_tokens,
                    error_message=f"{type(e).__name__}: {e}",
                    request_type=request_type,
                )

    async def run(
        self,
        prompt_token_ids_list: list,
        sampling_params_list: list,
        request_type_list: list,
        arrival_pattern: str,
        stagger_interval_sec: float,
        rate_rps: float,
        batch_size: int,
        batch_interval: float,
        seed: int = 42,
    ) -> tuple:
        """Execute all benchmark requests via HTTP and return (metrics_list, wall_start, wall_end)."""
        import httpx

        num_requests = len(prompt_token_ids_list)

        # Start embedded server if requested
        if self._server_embedded:
            self._start_embedded_server()
            print("Waiting for server to become ready...")
            ready = await self._wait_for_server()
            if not ready:
                self._stop_embedded_server()
                raise RuntimeError(
                    f"Embedded server did not become ready on port {self._port} "
                    f"within 120 seconds."
                )
            print("Server is ready.")
        else:
            # External server: verify it's reachable before sending requests
            print(f"Checking server at {self._base_url} ...")
            ready = await self._wait_for_server()
            if not ready:
                raise RuntimeError(
                    f"Server at {self._base_url} is not responding.\n"
                    f"Start it with: python -m babyvllm.entrypoints.cli "
                    f"--model <path> --port {self._port}"
                )
            print("Server is reachable.")

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )

        wall_start = time.perf_counter()

        raw_results = await _launch_with_arrival(
            self,
            num_requests=num_requests,
            prompt_token_ids_list=prompt_token_ids_list,
            sampling_params_list=sampling_params_list,
            request_type_list=request_type_list,
            arrival_pattern=arrival_pattern,
            stagger_interval_sec=stagger_interval_sec,
            rate_rps=rate_rps,
            batch_size=batch_size,
            batch_interval=batch_interval,
            seed=seed,
        )

        wall_end = time.perf_counter()

        # Convert gather results to PerRequestMetrics
        per_request_list = []
        for i, result in enumerate(raw_results):
            if isinstance(result, PerRequestMetrics):
                per_request_list.append(result)
            elif isinstance(result, Exception):
                per_request_list.append(PerRequestMetrics(
                    request_id=i,
                    status="error",
                    submit_time=0.0,
                    prompt_tokens=len(prompt_token_ids_list[i]),
                    error_message=f"Unhandled exception: {result}",
                    request_type=request_type_list[i],
                ))
            else:
                per_request_list.append(PerRequestMetrics(
                    request_id=i,
                    status="error",
                    submit_time=0.0,
                    prompt_tokens=len(prompt_token_ids_list[i]),
                    error_message=f"Unexpected result type: {type(result)}",
                    request_type=request_type_list[i],
                ))

        self._engine_stats = await self.get_engine_stats()

        # Cleanup
        await self._client.aclose()
        self._client = None
        if self._server_embedded:
            self._stop_embedded_server()

        return per_request_list, wall_start, wall_end


# ===========================================================================
# Main Entry Point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the online benchmark."""
    parser = argparse.ArgumentParser(
        description="baby-vllm Online Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct engine mode (default)
  python online_bench.py --model /path/to/model

  # HTTP mode against external server
  python online_bench.py --model /path/to/model --mode http --base-url http://localhost:8000

  # HTTP mode with embedded server
  python online_bench.py --model /path/to/model --mode http --server-embedded --port 8001

  # Higher concurrency, burst pattern
  python online_bench.py --model /path/to/model --concurrency 64 --num-requests 512

  # Poisson arrival at avg 10 req/s (random intervals, best real-world simulation)
  python online_bench.py --model /path/to/model --arrival-pattern poisson --rate-rps 10

  # Continuous arrival at fixed 10 req/s intervals
  python online_bench.py --model /path/to/model --arrival-pattern continuous --rate-rps 10

  # Non-streaming mode
  python online_bench.py --model /path/to/model --no-stream
        """,
    )

    # ---- Model ----
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the model directory (local filesystem path).",
    )

    # ---- Mode ----
    parser.add_argument(
        "--mode", type=str, default="direct",
        choices=["direct", "http"],
        help="Test mode: 'direct' uses AsyncLLMEngine directly, "
             "'http' sends HTTP requests. Default: direct.",
    )
    parser.add_argument(
        "--stream", action="store_true", default=True,
        help="Use streaming mode (default: enabled).",
    )
    parser.add_argument(
        "--no-stream", action="store_false", dest="stream",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000",
        help="Server URL for HTTP mode. Default: http://localhost:8000.",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for embedded server in HTTP mode. Default: 8000.",
    )
    parser.add_argument(
        "--server-embedded", action="store_true", default=False,
        help="Spawn API server as subprocess (HTTP mode only).",
    )

    # ---- Benchmark Parameters ----
    parser.add_argument(
        "--scenario", type=str, default="realistic-decode",
        choices=list(SCENARIO_PRESETS.keys()),
        help="Benchmark scenario preset. 'realistic-decode' uses Poisson arrivals "
             "with moderate prompt lengths and long generations, targeting a "
             "decode-heavy but realistic online workload for max_model_len=8192. "
             "Explicit CLI arguments override preset values. Default: realistic-decode.",
    )
    parser.add_argument(
        "--num-requests", type=int, default=128,
        help="Total number of requests. Default: 128.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=32,
        help="Maximum concurrent requests. Default: 32.",
    )
    parser.add_argument(
        "--min-input-len", type=int, default=16,
        help="Minimum prompt tokens per request for random workload. Default: 16.",
    )
    parser.add_argument(
        "--max-input-len", type=int, default=1024,
        help="Maximum prompt tokens per request. Default: 1024.",
    )
    parser.add_argument(
        "--min-output-len", type=int, default=8,
        help="Minimum generated tokens per request for random workload. Default: 8.",
    )
    parser.add_argument(
        "--max-output-len", type=int, default=1024,
        help="Maximum generated tokens per request. Default: 1024.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=10000,
        help="Random token id range. Default: 10000.",
    )
    parser.add_argument(
        "--workload", type=str, default="mixed",
        choices=["random", "mixed"],
        help="Request length workload. Default: mixed long-prompt-heavy online workload.",
    )
    parser.add_argument(
        "--prompt-len-distribution", type=str, default="bimodal",
        choices=["bimodal", "uniform", "lognormal"],
        help="Prompt length distribution for --workload mixed. Default: bimodal.",
    )
    parser.add_argument(
        "--long-prompt-ratio", type=float, default=0.7,
        help="Fraction of long-prompt requests for bimodal mixed workload. Default: 0.7.",
    )
    parser.add_argument(
        "--short-input-len", type=int, default=256,
        help="Upper bound for short prompt length in mixed workload. Default: 256.",
    )
    parser.add_argument(
        "--long-input-len", type=int, default=3072,
        help="Upper bound for long prompt length in mixed workload. Default: 3072.",
    )
    parser.add_argument(
        "--short-output-len", type=int, default=256,
        help="Upper bound for short request output length in mixed workload. Default: 256.",
    )
    parser.add_argument(
        "--long-output-len", type=int, default=1024,
        help="Upper bound for long-prompt request output length in mixed workload. Default: 1024.",
    )
    parser.add_argument(
        "--arrival-pattern", type=str, default="poisson",
        choices=["burst", "stagger", "continuous", "batch", "poisson"],
        help="Request arrival pattern. 'poisson' uses exponential inter-arrival "
             "times at --rate-rps average rate (best for real-world simulation). "
             "'batch' submits --batch-size requests every --batch-interval seconds "
             "without waiting for previous batches to finish. "
             "Default: poisson.",
    )
    parser.add_argument(
        "--stagger-interval-ms", type=int, default=50,
        help="Delay between requests for stagger pattern (ms). Default: 50.",
    )
    parser.add_argument(
        "--rate-rps", type=float, default=10.0,
        help="Target request rate for continuous/poisson pattern (req/s). "
             "For poisson, this is the average rate (lambda). Default: 10.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of requests per batch for 'batch' arrival pattern. "
             "Default: 32.",
    )
    parser.add_argument(
        "--batch-interval", type=float, default=5.0,
        help="Delay between batches in seconds for 'batch' arrival pattern. "
             "Default: 5.0.",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-request timeout in seconds. Default: 300.",
    )

    # ---- Engine Configuration ----
    parser.add_argument(
        "--enforce-eager", action="store_true", default=False,
        help="Disable CUDA graph optimization.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of tensor parallel replicas. Default: 1.",
    )
    parser.add_argument(
        "--data-parallel-size", type=int, default=1,
        help="Number of data parallel online replicas for direct or embedded-server mode. Default: 1.",
    )
    parser.add_argument(
        "--device-ids",
        default=None,
        help="Optional comma-separated CUDA device ids for direct or embedded-server mode. Uses the first DP*TP ids.",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Maximum model context length for direct/embedded-server mode. "
             "For external HTTP mode, start the server with the same --max-model-len. "
             "Default: engine default.",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=16384,
        help="Maximum total tokens per batch. Default: 16384.",
    )
    parser.add_argument(
        "--max-num-sequences", type=int, default=512,
        help="Maximum concurrent sequences in scheduler. Default: 512.",
    )
    parser.add_argument(
        "--max-prefill-tokens-per-step", type=int, default=8192,
        help="Maximum total Prefill tokens scheduled in one logical step. Default: 8192.",
    )
    parser.add_argument(
        "--max-prefill-chunk-size", type=int, default=4096,
        help="Maximum Prefill chunk size for a single sequence. Default: 4096.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory for KV cache. Default: 0.9.",
    )

    # ---- Output ----
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to export JSON results.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print per-request progress.",
    )
    parser.add_argument(
        "--profile", action="store_true", default=False,
        help="Enable PyTorch profiler and export Chrome trace.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42.",
    )

    args = apply_scenario_preset(parser.parse_args())
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")
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
    if args.data_parallel_size <= 0:
        raise ValueError("--data-parallel-size must be positive.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if args.device_ids is not None:
        args.device_ids = parse_nonnegative_int_csv(args.device_ids, "--device-ids")
    return args


async def main() -> None:
    """Async main entry point: parse args, create runner, execute benchmark."""
    args = parse_args()

    print("=" * 66)
    print(" baby-vllm Online Inference Benchmark")
    print("=" * 66)
    print(f" Model            : {args.model}")
    print(f" Mode             : {args.mode}")
    print(f" Scenario         : {args.scenario}")
    print(f" Streaming        : {args.stream}")
    print(f" Requests         : {args.num_requests}")
    print(f" Concurrency      : {args.concurrency}")
    print(f" Input Len        : {args.min_input_len}..{args.max_input_len}")
    print(f" Output Len       : {args.min_output_len}..{args.max_output_len}")
    print(f" Max Model Len    : {args.max_model_len or 'engine default'}")
    print(f" Batch Tokens     : {args.max_num_batched_tokens}")
    print(f" Max Sequences    : {args.max_num_sequences}")
    print(f" Prefill Tokens   : {args.max_prefill_tokens_per_step}")
    print(f" Prefill Chunk    : {args.max_prefill_chunk_size}")
    print(f" Data Parallel    : {args.data_parallel_size}")
    print(f" Tensor Parallel  : {args.tensor_parallel_size}")
    if args.device_ids is not None:
        print(f" Device IDs       : {args.device_ids}")
    print(f" Workload         : {args.workload}")
    if args.workload == "mixed":
        print(f" Prompt Dist      : {args.prompt_len_distribution}")
        print(f" Long Prompt Ratio: {args.long_prompt_ratio:.2f}")
        print(f" Short/Long Input : {args.short_input_len} / {args.long_input_len}")
        print(f" Short/Long Output: {args.short_output_len} / {args.long_output_len}")
    print(f" Arrival Pattern  : {args.arrival_pattern}")
    if args.arrival_pattern == "batch":
        print(f" Batch Size       : {args.batch_size}")
        print(f" Batch Interval   : {args.batch_interval:.1f}s")
    elif args.arrival_pattern == "stagger":
        print(f" Stagger Interval : {args.stagger_interval_ms}ms")
    elif args.arrival_pattern in ("continuous", "poisson"):
        suffix = " (random intervals)" if args.arrival_pattern == "poisson" else ""
        print(f" Target Rate      : {args.rate_rps} req/s{suffix}")
    print(f" Seed             : {args.seed}")
    print("=" * 66)

    # Generate test data
    if args.workload == "mixed":
        prompt_token_ids_list, sampling_params_list, request_type_list = generate_mixed_test_data(
            num_requests=args.num_requests,
            prompt_len_distribution=args.prompt_len_distribution,
            short_input_len=args.short_input_len,
            long_input_len=args.long_input_len,
            short_output_len=args.short_output_len,
            long_output_len=args.long_output_len,
            long_prompt_ratio=args.long_prompt_ratio,
            seed_val=args.seed,
        )
    else:
        prompt_token_ids_list, sampling_params_list, request_type_list = generate_random_test_data(
            num_requests=args.num_requests,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
            min_output_len=args.min_output_len,
            max_output_len=args.max_output_len,
            vocab_size=args.vocab_size,
            seed_val=args.seed,
        )
    total_prompt = sum(len(p) for p in prompt_token_ids_list)
    total_max_gen = sum(sp.max_tokens for sp in sampling_params_list)
    type_counts = {t: request_type_list.count(t) for t in sorted(set(request_type_list))}
    print(f"\nGenerated {args.num_requests} requests: "
          f"{total_prompt:,} total prompt tokens, "
          f"up to {total_max_gen:,} generation tokens")
    print(f"Request types: {type_counts}")

    selected_device_ids = select_device_ids(
        args.device_ids,
        dp_size=args.data_parallel_size,
        tp_size=args.tensor_parallel_size,
    )
    if args.mode == "direct" or args.server_embedded:
        preflight_cuda_devices(
            dp_size=args.data_parallel_size,
            tp_size=args.tensor_parallel_size,
            device_ids=selected_device_ids,
        )

    # Build engine kwargs (only used for direct mode or embedded server)
    engine_kwargs = {
        "enforce_eager": args.enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "data_parallel_device_ids": selected_device_ids,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_sequences": args.max_num_sequences,
        "max_prefill_tokens_per_step": args.max_prefill_tokens_per_step,
        "max_prefill_chunk_size": args.max_prefill_chunk_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_length"] = args.max_model_len

    # Create runner
    if args.mode == "direct":
        runner = DirectEngineRunner(
            model=args.model,
            engine_kwargs=engine_kwargs,
            concurrency=args.concurrency,
            timeout=args.timeout,
            verbose=args.verbose,
        )
    else:  # http
        runner = HTTPAPIRunner(
            model=args.model,
            base_url=args.base_url,
            port=args.port,
            server_embedded=args.server_embedded,
            engine_kwargs=engine_kwargs,
            stream=args.stream,
            concurrency=args.concurrency,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    # Stagger interval in seconds
    stagger_interval_sec = args.stagger_interval_ms / 1000.0

    # Start GPU sampler
    gpu_sampler = GPUUsageSampler()
    await gpu_sampler.start()

    # Run benchmark (with optional PyTorch profiler)
    run_fn = lambda: runner.run(
        prompt_token_ids_list,
        sampling_params_list,
        request_type_list,
        args.arrival_pattern,
        stagger_interval_sec,
        args.rate_rps,
        args.batch_size,
        args.batch_interval,
        args.seed,
    )

    if args.profile:
        print("\nPyTorch profiler enabled. Exporting trace to trace_online_baby.json...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            per_request_list, wall_start, wall_end = await run_fn()
        prof.export_chrome_trace("trace_online_baby.json")
        print("Profiler trace exported.")
    else:
        per_request_list, wall_start, wall_end = await run_fn()

    # Stop GPU sampler
    await gpu_sampler.stop()

    # Compute aggregate metrics
    agg = compute_aggregate_metrics(
        per_request_list,
        wall_start,
        wall_end,
        avg_gpu_memory_mb=gpu_sampler.avg_memory_mb,
        avg_gpu_utilization=gpu_sampler.avg_utilization,
    )

    engine_stats = await runner.get_engine_stats()

    # Print report
    print_report(args, agg)
    print_request_type_breakdown(per_request_list)
    print_engine_stats(engine_stats)

    # Export JSON if requested
    if args.output:
        export_json(
            agg,
            per_request_list,
            args.output,
            config=args,
            engine_stats=engine_stats,
        )


if __name__ == "__main__":
    asyncio.run(main())
