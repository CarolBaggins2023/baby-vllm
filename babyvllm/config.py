import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_sequences: int = 512
    max_model_length: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    host: str = "0.0.0.0"
    port: int = 8000
    # Store some configs in huggingface' config.
    hf_config: AutoConfig | None = None
    
    def __post_init__(self):
        # Normalize the model path: expand ~ and convert to absolute path early.
        self.model = os.path.abspath(os.path.expanduser(self.model))

        if not os.path.isdir(self.model):
            # Build a rich diagnostic message.
            parts = [
                f"Model path does not exist or is not a directory.",
                f"  model (resolved): {self.model}",
                f"  current working directory: {os.getcwd()}",
            ]

            home_dir = os.path.expanduser("~")
            # Check if the parent of the resolved path exists; if so, list
            # siblings to help the user spot a typo.
            parent = os.path.dirname(self.model)
            if os.path.isdir(parent):
                siblings = sorted(os.listdir(parent))[:20]
                parts.append(f"  parent directory '{parent}' exists; contents (first 20): {siblings}")
            else:
                parts.append(f"  parent directory '{parent}' does NOT exist either")

            # Reconstruct the relative path (from CWD) and check if it
            # exists under $HOME instead — common when user forgets a
            # leading slash on an absolute path like /root/autodl-tmp/...
            rel = os.path.relpath(self.model, os.getcwd())
            home_candidate = os.path.join(home_dir, rel)
            if os.path.isdir(home_candidate):
                parts.append(f"  DID find model under $HOME: {home_candidate}")
                parts.append(f"  → try: BABYVLLM_TEST_MODEL_PATH={home_candidate}")

            parts.append(
                f"  Hint: set BABYVLLM_TEST_MODEL_PATH to an absolute path, e.g.\n"
                f"        BABYVLLM_TEST_MODEL_PATH=/root/autodl-tmp/Qwen/Qwen3-0.6B"
            )
            raise ValueError("\n".join(parts))
        # Due to the use of the "flash_dattn",
        # the block size of KV cache must be divisible by 256.
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_length = min(self.max_model_length, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_length
        assert isinstance(self.host, str) and len(self.host) > 0
        assert isinstance(self.port, int) and 1 <= self.port <= 65535
    