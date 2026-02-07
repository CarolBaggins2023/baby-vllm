from enum import Enum, auto
from itertools import count
from copy import copy
import math

from babyvllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
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
    def if_finished(self):
        return self.status == SequenceStatus.FINISHED
    
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
    
    def __getstate__(self):
        # If number of completion tokens is 0, then the sequence is in prefill phase.
        # In prefill phase, returns the whole token ids.
        # In decode phase, returns only the last token id.
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )
    
    def __setstate__(self, state):
        # `state` should be a tuple of
        # (num_tokens, num_prompt_tokens, num_cached_tokens, block_table, last_token or token_ids)
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        
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
    