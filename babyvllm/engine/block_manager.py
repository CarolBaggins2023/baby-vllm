import xxhash
from collections import deque
import numpy as np
import math

from babyvllm.engine.sequence import Sequence

class Block:
    """
    A block of token ids with fixed size, which is used to store kv cache.
    """
    
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.token_ids = []
        # The hash value of the block.
        # If hash value is -1, then the block is invalid.
        self.hash = -1
        # The number of references to the block.
        # If reference count is 0, then no sequence is using the block.
        # It works like the reference count of shared pointer in C++.
        self.ref_count = 0
    
    def update(self, h: int, token_ids: list[int]):
        self.hash = h
        self.token_ids = token_ids
    
    def reset(self):
        self.token_ids = []
        self.hash = -1
        self.ref_count = 1
        
class BlockManager:
    """
    Manage the blocks of token id sequences. Responsible for allocating and deallocating blocks, and 
    maintaining the hash value to block id mapping.
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        # The number of tokens in each block.
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        
        # The hash value to block id mapping.
        # In prefix caching, partial blocks are merely impossible to be hit,
        # which results in two problems:
        # (1) Waste of memory;
        # (2) Increase hash collision rate.
        # So, only fully filled blocks are recorded in `hash_to_block_id`.
        self.hash_to_block_id = {}
        
        # Blocks in `free_block_ids` are not used by any sequence.
        # Blocks in `used_block_ids` are used by at least one sequence.
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
        
        # The hash value to block id mapping is decoupled from the block state.
        # `hash_to_block_id` records which block used to store which segment of token ids.
        # `free_block_ids` and `used_block_ids` records whether the block is used by any sequence.
        # If one block used to store a segement of token ids, but it is currently not used by that segment,
        # then the id of this block is in `hash_to_block_id`, but not in `used_block_ids`.
    
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix_hash_value: int) -> int:
        """
        Compute the hash value of block.
        Args:
            token_ids: The token ids in the block to compute hash value.
            prefix_hash_value: The prefix hash value of previous block. If -1, then no previous block.
        
        Returns:
            The hash value of block.
        """
        
        # `h` is the initial hash object.
        h = xxhash.xxh64()
        # If there is a previous block, then update the hash object with the hash value of previous block.
        if prefix_hash_value != -1:
            # Update the hash object with the hash value of previous block.
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        # Update the hash object with the token ids in the block.
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        
        # Output the integer digest of hash object.
        # Integer digest is more convenient to store and compare than the byte stream.
        return h.intdigest()
    
    def _allocate_block(self, block_id: int) -> Block:
        """
        Allocate a block with given block id. And manage the free and used block ids.
        Args:
            block_id: The id of block to allocate.
        
        Returns:
            The allocated block.
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already allocated."
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _deallocate_block(self, block_id: int):
        """
        Deallocate a block with given block id. And manage the free and used block ids.
        Args:
            block_id: The id of block to deallocate.
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} cannot be deallocated because it is referenced by {block.ref_count} seqeuences."
        block.token_ids = []
        self.free_block_ids.append(block_id)
        self.used_block_ids.remove(block_id)
    
    def deallocate(self, seq: Sequence):
        # Update block information.
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        # Update sequence information.
        seq.block_table = []
        seq.num_cached_tokens = 0

    def can_allocate_chunk(self, seq: Sequence, chunk_size: int) -> bool:
        """ Whether there is enough free blocks to accommodate `chunk_size` tokens. """
        
        current_blocks = len(seq.block_table)
        target_blocks = math.ceil((seq.num_computed_tokens+chunk_size)/self.block_size)
        num_new_blocks_needed = target_blocks-current_blocks
        
        return num_new_blocks_needed <= len(self.free_block_ids)
    
    def allocate_chunk(self, seq: Sequence, chunk_size: int):
        """ Allocate physical blocks of `chunk_size` tokens to the sequence. """
        
        # Token index range in the sequence.
        start = seq.num_computed_tokens
        end = start+chunk_size
        
        # Which logical blocks are needed to compute this chunk of tokens.
        first_block_idx = start//self.block_size
        last_block_idx = (end-1)//self.block_size
        
        # Allocate physical blocks.
        for i in range(first_block_idx, last_block_idx+1):
            # Whether we need to apply a new physical block.
            is_new_block = (i>=len(seq.block_table))
            
            # Whether this block will be fully filled.
            # Only fully filled blocks will be managed in hash table,
            # and participate in prefix caching.
            is_full = (end>=(i+1)*self.block_size)
            
            block_start_token = i*self.block_size
            block_end_token = min((i+1)*self.block_size, end)
            token_ids = seq.token_ids[block_start_token:block_end_token]
            
            # ----- Compute Prefix Hash -----
            if is_full:
                # Chain hash: The hash of the current block depends on the hash of the previous block.
                prefix_hash = self.blocks[seq.block_table[i-1]].hash if i > 0 else -1
                h = self.compute_hash(token_ids=token_ids, prefix_hash_value=prefix_hash)
            else:
                h = -1
            
            # ----- Allocate and write physical block. -----
            if is_new_block:
                # If it is a new physical block, we need to execute complete
                # allocation and cache reuse logic.
                if is_full:
                    # If the block is full, we need to execute cache reuse.
                    # Fetch the block id from hash value to block id mapping.
                    # If the hash value is not found, then set the block id to -1, which means cache miss.
                    block_id = self.hash_to_block_id.get(h, -1)
                    cache_collision = (block_id != -1) and (self.blocks[block_id].token_ids != token_ids)
                    # `block_id == -1` means cache miss.
                    # Both cache miss and hash collision means no cache found.
                    no_cache_found = (block_id == -1) or cache_collision
                    
                    if not no_cache_found:
                        # Cache hit.
                        seq.num_cached_tokens += self.block_size
                        block = self.blocks[block_id]
                        if block_id in self.used_block_ids:
                            block.ref_count += 1
                        else:
                            block = self._allocate_block(block_id)
                    else:
                        # Cache miss.
                        block = self._allocate_block(self.free_block_ids[0])
                        block.update(h, token_ids)
                        self.hash_to_block_id[h] = block.block_id
                else:
                    # If the block is not full, we do not need to execute cache reuse.
                    block = self._allocate_block(self.free_block_ids[0])
                    block.update(h, token_ids)
                
                seq.block_table.append(block.block_id)
            else:
                # If it is not a new physical block, we just need to append new token.
                block = self.blocks[seq.block_table[i]]
                block.update(h, token_ids)
                # If this append operation will make the block full, register the block id in hash table.
                if is_full and h != -1:
                    self.hash_to_block_id[h] = block.block_id
