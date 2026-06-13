from __future__ import annotations

import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_sequences: int = 512
    max_model_length: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # Number of full model replicas managed by a parent DP coordinator.
    data_parallel_size: int = 1
    # This process's DP rank within the full DP world.
    data_parallel_rank: int = 0
    # Total DP world size. DP workers set data_parallel_size=1 internally but
    # keep this value so device mapping still sees the full DP layout.
    data_parallel_world_size: int | None = None
    # Base TCP port used to derive per-DP-rank TP process-group endpoints.
    data_parallel_base_port: int = 12345
    # Optional physical CUDA device order for all DP*TP logical ranks.
    data_parallel_device_ids: list[int] | None = None
    # Torch distributed init endpoint for the TP group inside this DP replica.
    distributed_init_method: str | None = None
    # Shared-memory segment name used by TP workers inside this DP replica.
    shared_memory_name: str | None = None
    enforce_eager: bool = False
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Store some configs in huggingface' config.
    hf_config: AutoConfig | None = None
    
    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"Model path does not exist or is not a directory: {self.model}")
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be divisible by 256.")
        self._validate_parallel_sizes()
        self._validate_device_mapping()
        if self.distributed_init_method is None:
            self.distributed_init_method = self._default_distributed_init_method()
        if self.shared_memory_name is None:
            self.shared_memory_name = self._default_shared_memory_name()
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_length = min(self.max_model_length, self.hf_config.max_position_embeddings)
        self._validate_qwen3_tensor_parallel_config()
        if self.max_num_batched_tokens < self.max_model_length:
            raise ValueError("max_num_batched_tokens must be at least max_model_length.")

    @property
    def effective_data_parallel_size(self) -> int:
        return self.data_parallel_world_size or self.data_parallel_size

    def device_id_for_rank(self, tp_rank: int) -> int:
        """Map a local TP rank in this DP replica to a physical CUDA device.

        By default, DP rank d and TP rank t use device d*tensor_parallel_size+t.
        If data_parallel_device_ids is provided, that list defines the physical
        device order for all DP*TP logical ranks.
        """
        if not isinstance(tp_rank, int) or isinstance(tp_rank, bool):
            raise ValueError("tp_rank must be an integer.")
        if not 0 <= tp_rank < self.tensor_parallel_size:
            raise ValueError(
                f"tp_rank ({tp_rank}) must be in [0, {self.tensor_parallel_size})."
            )
        if self.data_parallel_device_ids is not None:
            offset = self.data_parallel_rank*self.tensor_parallel_size+tp_rank
            return self.data_parallel_device_ids[offset]
        return self.data_parallel_rank*self.tensor_parallel_size+tp_rank

    def _validate_parallel_sizes(self):
        for name in ("tensor_parallel_size", "data_parallel_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.tensor_parallel_size > 8:
            raise ValueError("tensor_parallel_size must be <= 8.")
        if self.data_parallel_world_size is None:
            self.data_parallel_world_size = self.data_parallel_size
        if (
            not isinstance(self.data_parallel_world_size, int)
            or isinstance(self.data_parallel_world_size, bool)
            or self.data_parallel_world_size <= 0
        ):
            raise ValueError("data_parallel_world_size must be a positive integer.")
        if not isinstance(self.data_parallel_rank, int) or isinstance(self.data_parallel_rank, bool):
            raise ValueError("data_parallel_rank must be an integer.")
        if not 0 <= self.data_parallel_rank < self.effective_data_parallel_size:
            raise ValueError(
                f"data_parallel_rank ({self.data_parallel_rank}) must be in "
                f"[0, {self.effective_data_parallel_size})."
            )
        if (
            not isinstance(self.data_parallel_base_port, int)
            or isinstance(self.data_parallel_base_port, bool)
            or self.data_parallel_base_port <= 0
        ):
            raise ValueError("data_parallel_base_port must be a positive integer.")

    def _validate_device_mapping(self):
        required_devices = self.effective_data_parallel_size*self.tensor_parallel_size
        if self.data_parallel_device_ids is not None:
            if not isinstance(self.data_parallel_device_ids, list):
                raise ValueError("data_parallel_device_ids must be a list[int] or None.")
            if len(self.data_parallel_device_ids) != required_devices:
                raise ValueError(
                    "data_parallel_device_ids length must equal "
                    "data_parallel_world_size*tensor_parallel_size."
                )
            if any(
                not isinstance(device_id, int)
                or isinstance(device_id, bool)
                or device_id < 0
                for device_id in self.data_parallel_device_ids
            ):
                raise ValueError("data_parallel_device_ids must contain non-negative integers.")
            if len(set(self.data_parallel_device_ids)) != len(self.data_parallel_device_ids):
                raise ValueError("data_parallel_device_ids must contain unique device ids.")

        if required_devices <= 1 and self.data_parallel_device_ids is None:
            return

        cuda_device_count = torch.cuda.device_count()
        if self.data_parallel_device_ids is None:
            if cuda_device_count < required_devices:
                raise ValueError(
                    f"data_parallel_size*tensor_parallel_size requires {required_devices} "
                    f"CUDA devices, but only {cuda_device_count} are available."
                )
            return

        max_device_id = max(self.data_parallel_device_ids, default=-1)
        if cuda_device_count <= max_device_id:
            raise ValueError(
                f"data_parallel_device_ids references cuda:{max_device_id}, but only "
                f"{cuda_device_count} CUDA devices are available."
            )

    def _default_distributed_init_method(self) -> str:
        return f"tcp://localhost:{self.data_parallel_base_port+self.data_parallel_rank}"

    def _default_shared_memory_name(self) -> str:
        return f"babyvllm_dp{self.data_parallel_rank}_{os.getpid()}"

    def _validate_qwen3_tensor_parallel_config(self):
        tp_size = self.tensor_parallel_size
        num_attention_heads = self._require_positive_int_config("num_attention_heads")
        num_key_value_heads = getattr(self.hf_config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        elif not isinstance(num_key_value_heads, int) or isinstance(num_key_value_heads, bool) or num_key_value_heads <= 0:
            raise ValueError("hf_config.num_key_value_heads must be a positive integer when set.")
        intermediate_size = self._require_positive_int_config("intermediate_size")

        local_num_heads = self._validate_tp_divisibility(
            field_name="num_attention_heads",
            value=num_attention_heads,
            tp_size=tp_size,
        )
        local_num_kv_heads = self._validate_tp_divisibility(
            field_name="num_key_value_heads",
            value=num_key_value_heads,
            tp_size=tp_size,
            extra=(
                "KV-head replication is not supported by baby-vllm-basic; "
                f"effective num_key_value_heads={num_key_value_heads}, "
                f"tensor_parallel_size={tp_size}."
            ),
        )
        local_intermediate_size = self._validate_tp_divisibility(
            field_name="intermediate_size",
            value=intermediate_size,
            tp_size=tp_size,
        )

        if local_num_heads <= 0:
            raise ValueError(
                "tensor_parallel_size produces zero local attention heads: "
                f"num_attention_heads={num_attention_heads}, tensor_parallel_size={tp_size}."
            )
        if local_num_kv_heads <= 0:
            raise ValueError(
                "tensor_parallel_size produces zero local KV heads: "
                f"effective num_key_value_heads={num_key_value_heads}, "
                f"tensor_parallel_size={tp_size}. KV-head replication is not "
                "supported by baby-vllm-basic."
            )
        if local_intermediate_size <= 0:
            raise ValueError(
                "tensor_parallel_size produces zero local MLP features: "
                f"intermediate_size={intermediate_size}, tensor_parallel_size={tp_size}."
            )

    def _require_positive_int_config(self, field_name: str) -> int:
        value = getattr(self.hf_config, field_name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"hf_config.{field_name} must be a positive integer.")
        return value

    def _validate_tp_divisibility(
        self,
        *,
        field_name: str,
        value: int,
        tp_size: int,
        extra: str | None = None,
    ) -> int:
        if value % tp_size != 0:
            message = (
                f"hf_config.{field_name}={value} must be divisible by "
                f"tensor_parallel_size={tp_size}."
            )
            if extra is not None:
                message = f"{message} {extra}"
            raise ValueError(message)
        return value // tp_size
    
