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
    # WHY: Server host/port belong in Config (not just CLI args) so they serve as
    # the single source of truth.  CLI args flow through engine_kwargs → Config,
    # and then cli.py reads them back from engine.engine.config when launching
    # uvicorn.  This avoids two independent sources of truth (CLI args + config)
    # that could diverge.
    #
    # Default is "0.0.0.0" (bind all interfaces), which is the permissive
    # Config-level default.  The CLI --host default is "127.0.0.1" (localhost
    # only) for security — the CLI default overrides this via engine_kwargs.
    #
    # Example: a script that creates Config(model="...", port=9999) directly
    # (bypassing CLI) will bind to 0.0.0.0:9999 by default, which is the
    # expected behaviour for library usage.
    host: str = "0.0.0.0"
    port: int = 8000
    # Store some configs in huggingface' config.
    hf_config: AutoConfig | None = None
    
    def __post_init__(self):
        assert os.path.isdir(self.model), (
            f"Model path '{self.model}' does not exist or is not a directory. "
            f"Please check that the path is correct and the model files are present."
        )
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_length = min(self.max_model_length, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_length
        # WHY: Validate host/port early (in __post_init__) so invalid values are
        # caught before GPU memory is allocated.  This fails fast instead of
        # letting uvicorn raise a cryptic error 30 seconds into startup.
        #
        # Example: if a script passes port=99999, this assertion fires immediately
        # with a clear message, rather than uvicorn failing with
        # "port must be 0-65535" after the model has already been loaded.
        assert isinstance(self.host, str) and len(self.host) > 0, (
            f"host must be a non-empty string, got {self.host!r}"
        )
        assert isinstance(self.port, int) and 1 <= self.port <= 65535, (
            f"port must be between 1 and 65535, got {self.port}"
        )
