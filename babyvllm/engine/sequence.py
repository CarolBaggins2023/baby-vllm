from __future__ import annotations
from enum import Enum, auto
from itertools import count
from copy import copy
import math
from typing import Any

from babyvllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    # The sequence is chunking the prompt, and generation has not yet begun.
    CHUNKED_PREFILL = auto()

class Sequence:
    """Sequence of tokens. It also records the related cached blocks.
    """
    
    # shared class property
    # maximum number of tokens in a sequence
    block_size = 256
    # global index generator for sequences
    # It will generate increment unique sequence id.
    counter = count()
    
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # unique sequence id
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        
        self.token_ids = copy(token_ids)
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.num_tokens = len(self.token_ids)
        # When the sequence is created, it is filled with prompt tokens.
        self.num_prompt_tokens = len(self.token_ids)
        # Number of tokens that have been processed through model forward propogation,
        # and written to the kv cache.
        self.num_computed_tokens = 0
        # Record the amount of token budget allocated in the current scheduling step
        self.chunk_size = 0
        
        # `block_table` stores the physical block ids assigned by BlockManager to this sequence.
        # For example, if the sequence is divided into 3 logical blocks,
        # and BlockManager places the kv cache of these logical blocks in the 10th, 45th, and 2nd
        # blocks of the physical memory pool, then the `block_table` of this sequence will be [10, 45, 2].
        # Here, the physical memory pool is the `allocated_kv_cache` assigned in `ModelRunner.allocate_kv_cache()`.
        self.num_cached_tokens = 0
        self.block_table = []
        
        # Sampling parameters for the sequence.
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length
    
    def __len__(self):
        return self.num_tokens
    
    def __getitem__(self, idx):
        return self.token_ids[idx]
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def is_chunked_prefill(self):
        """ Whether the sequence is chunking the prompt, and generation has not yet begun. """
        return 0 < self.num_computed_tokens < self.num_prompt_tokens
    
    @property
    def get_uncomputed_token_ids(self):
        if self.num_computed_tokens < self.num_prompt_tokens:
            return self.token_ids[self.num_computed_tokens:self.num_prompt_tokens]
        return []
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens-self.num_prompt_tokens
    
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_blocks(self):
        return math.ceil(self.num_tokens/self.block_size)

    @property    
    def num_cached_blocks(self):
        return math.ceil(self.num_cached_tokens/self.block_size)
    
    @property
    def last_block_num_tokens(self):
        """ Calculate the number of tokens in the last block. """
        return self.num_tokens-(self.num_blocks-1)*self.block_size
    
    def block(self, i: int):
        """ Get the i-th block. """
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        # In python, slice logic automatically clips the indices to the length of the list.
        # So, "(i+1)*self.block_size" just works.
        return self.token_ids[i*self.block_size:(i+1)*self.block_size]
    
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def _is_decode_worker_state(self) -> bool:
        return (
            self.chunk_size == 1
            and self.num_computed_tokens > 0
            and self.num_computed_tokens == self.num_tokens-1
        )

    def prefill_uncached_token_count(self) -> int:
        return len(self.token_ids)-self.num_cached_tokens

    def validate_prefill_state(self) -> int:
        num_uncached_tokens = self.prefill_uncached_token_count()
        if num_uncached_tokens <= 0:
            seq_id = getattr(self, "seq_id", "?")
            raise ValueError(
                f"Sequence {seq_id} has no uncached query tokens for prefill "
                f"(token_ids={len(self.token_ids)}, "
                f"num_cached_tokens={self.num_cached_tokens}). "
                "Prefix cache must leave at least one token uncached."
            )
        return num_uncached_tokens

    def to_worker_state(self) -> dict[str, Any]:
        """Build a chunk-aware, pickle-friendly state for TP worker ranks."""
        is_decode = self._is_decode_worker_state()
        return {
            "version": 2,
            "phase": "decode" if is_decode else "prefill",
            "seq_id": self.seq_id,
            "status": self.status.name,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "num_computed_tokens": self.num_computed_tokens,
            "chunk_size": self.chunk_size,
            "block_table": list(self.block_table),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "max_model_length": self.max_model_length,
            "last_token": self.last_token,
            # Main's prepare_forward slices token_ids by absolute token index.
            # Decode workers only need the current token, so reconstruct with
            # placeholders on the receiving side to keep indexing stable.
            "token_ids": None if is_decode else list(self.token_ids),
        }

    @classmethod
    def from_worker_state(cls, state: dict[str, Any]) -> "Sequence":
        phase = state.get("phase")
        num_tokens = state["num_tokens"]
        if phase == "decode":
            token_ids = [0]*num_tokens
            if num_tokens > 0:
                token_ids[state["num_computed_tokens"]] = state["last_token"]
        elif phase == "prefill":
            token_ids = list(state["token_ids"])
            if len(token_ids) != num_tokens:
                raise ValueError(
                    "Prefill sequence worker state must include full token_ids "
                    f"(got {len(token_ids)}, expected {num_tokens})."
                )
        else:
            raise ValueError(f"Unknown sequence worker-state phase: {phase!r}")

        seq = cls.__new__(cls)
        seq.seq_id = state["seq_id"]
        status = state.get("status", "WAITING")
        seq.status = SequenceStatus[status] if isinstance(status, str) else status
        seq.token_ids = token_ids
        seq.last_token = state["last_token"]
        seq.num_tokens = num_tokens
        seq.num_prompt_tokens = state["num_prompt_tokens"]
        seq.num_cached_tokens = state["num_cached_tokens"]
        seq.num_computed_tokens = state["num_computed_tokens"]
        seq.chunk_size = state["chunk_size"]
        seq.block_table = list(state["block_table"])
        seq.temperature = state["temperature"]
        seq.max_tokens = state["max_tokens"]
        seq.ignore_eos = state["ignore_eos"]
        seq.max_model_length = state["max_model_length"]
        return seq
    
    def __getstate__(self):
        return self.to_worker_state()
    
    def __setstate__(self, state):
        if isinstance(state, dict):
            restored = self.from_worker_state(state)
            self.__dict__.update(restored.__dict__)
            return

        # Strictly unpack in the order of `__getstate__`.
        (self.num_tokens,
         self.num_prompt_tokens,
         self.num_cached_tokens,
         self.num_computed_tokens,
         self.chunk_size,
         self.block_table) = state[:-1]
        
        # In prefill phase, all token ids should be restored.
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        # In decode phase, only the last token id need to be restored.
        # Because other token ids are already cached in the KV cache, stored in `block_table`,
        # it is not necessary to store other token ids, for saving memory.
        else:
            self.token_ids = [state[-1]]
        
        # Update the last token id.
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.temperature = 1.0
        self.max_tokens = 64
        self.ignore_eos = False
        self.max_model_length = None
    
