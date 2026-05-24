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
        if not os.path.isdir(self.model):
            raise ValueError(
                f"Model path does not exist or is not a directory.\n"
                f"  model (as provided): {self.model}\n"
                f"  resolved absolute path: {os.path.abspath(self.model)}\n"
                f"  current working directory: {os.getcwd()}\n"
                f"  Hint: set BABYVLLM_TEST_MODEL_PATH env var to the correct model directory."
            )
        # Due to the use of the "flash_dattn",
        # the block size of KV cache must be divisible by 256.
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_length = min(self.max_model_length, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_length
        assert isinstance(self.host, str) and len(self.host) > 0
        assert isinstance(self.port, int) and 1 <= self.port <= 65535
    