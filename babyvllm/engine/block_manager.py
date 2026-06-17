from __future__ import annotations

import xxhash
from collections import deque
import numpy as np

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
    
    def can_allocate(self, seq: Sequence) -> bool:
        """
        Check whether the block manager can allocate `num_blocks` blocks for the sequence.
        """
        return seq.num_blocks <= len(self.free_block_ids)
    
    def allocate(self, seq: Sequence):
        # Initial hash value. -1 means no previous block.
        h = -1
        # The sequence needs `num_blocks` blocks to store all tokens,
        # so the block manager needs to allocate `num_blocks` blocks for the sequence.
        for i in range(seq.num_blocks):
            # Get the token ids in range of i-th block of the sequence.
            token_ids = seq.block(i)
            # If the block is full, then compute the hash value of the block.
            # Otherwise, set the hash value of the block to -1.
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            # Fetch the block id from hash value to block id mapping.
            # If the hash value is not found, then set the block id to -1, which means cache miss.
            block_id = self.hash_to_block_id.get(h, -1)
            
            # `no_cache_found == False` means an existing block can be reused.
            # `no_cache_found == True` means a new block needs to be allocated.
            no_cache_found = False
            # `block_id == -1` means cache miss.
            # `self.blocks[block_id].token_ids != token_ids` means hash collision.
            # Both cache miss and hash collision means no cache found.
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True
            
            if not no_cache_found:
                # Update sequence information.
                seq.num_cached_tokens += self.block_size
                
                # Update block information.
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                # The block used to store this segment of token ids, but now it is not allocated.
                else:
                    block = self._allocate_block(block_id)
            else:
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h, token_ids)
                # If the block is fulled filled, then record its hash value.
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            
            # Record which blocks are used by the sequence.
            # This information will be used to free blocks or append tokens to the block.
            seq.block_table.append(block.block_id)
    
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
    
    def can_append(self, seq: Sequence) -> bool:
        """ Whether we can append a new token to the sequence. """
        
        # If the sequence requires a new cache block, we need to check whether there are free blocks.
        if seq.num_tokens%self.block_size == 1:
            return len(self.free_block_ids) > 0
        # Otherwise, we can append a token to the sequence.
        return True
    
    def append(self, seq: Sequence):
        """
        Manage the blocks of sequence when a new token is appended to the sequence.
        There are three cases:
        (1) The last cache block of the sequence used to have `block_size`-1 tokens, so its hash value is -1.
            After a token is appended, the last cache block of the sequence will have `block_size` tokens, so it will have a valid hash value
            and its hash value should be managed in hash table.
        (2) The last cache block of the seqeunce was already fully filled.
            After a token is appended, the last cache block will have `block_size`+1 tokens, which is out of limit.
            So, a new cache block is needed to store the appended token of the sequence.
        (3) The number of tokens in the last cache block of the sequence is less than `block_size`-1.
            After a token is appended, the last cache block is still unfully filled.
            So, the hash value of the last cache block remains -1 and the hash mapping remains not recorded.
        """
        
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if seq.num_tokens%self.block_size == 0:
            h = self.compute_hash(token_ids=seq.block(seq.num_blocks-1), prefix_hash_value=-1 if len(block_table) == 1 else self.blocks[block_table[-2]].hash)
            last_block.update(h=h, token_ids=seq.block(seq.num_blocks-1))
            self.hash_to_block_id[h] = last_block.block_id
        elif seq.num_tokens%self.block_size == 1:
            appended_block = self._allocate_block(self.free_block_ids[0])
            block_table.append(appended_block.block_id)
        else:
            assert last_block.hash == -1
            
