import xxhash
from collections import deque
import numpy as np

from babyvllm.engine.sequence import Sequence

class Block:
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
        self.ref_count = 0
        
class BlockManager:
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
    
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        """
        Compute the hash value of block.
        Args:
            token_ids: The token ids in the block to compute hash value.
            prefix_hash_value: The prefix hash value of previous block. If -1, then no previous block.
        
        Returns:
            The hash value of block.
        """
        
        # `h` is the initial hash object.
        h = xxhash.xx64()
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
    
    def allocate(self, seq: Sequence) -> None:
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
            