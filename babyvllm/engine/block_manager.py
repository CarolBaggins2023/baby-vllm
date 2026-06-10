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
        self.token_ids = list(token_ids)
    
    def clear_metadata(self):
        self.token_ids = []
        self.hash = -1

    def reset(self):
        self.clear_metadata()
        self.ref_count = 0
        
class BlockManager:
    """
    Manage the blocks of token id sequences. Responsible for allocating and deallocating blocks, and 
    maintaining the hash value to block id mapping.
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        # The number of tokens in each block.
        self.num_blocks = num_blocks
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
    
    def _remove_hash_mapping(self, block: Block):
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]

    def _move_free_to_used(self, block_id: int):
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)

    def _allocate_block(self, block_id: int) -> Block:
        """
        Allocate a free block for new content and clear any stale cache metadata.
        Args:
            block_id: The id of block to allocate.
        
        Returns:
            The allocated block.
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already allocated."
        self._remove_hash_mapping(block)
        block.clear_metadata()
        block.ref_count = 1
        self._move_free_to_used(block_id)
        return block

    def _activate_cached_block(self, block_id: int) -> Block:
        """
        Move an unreferenced cached block back to used state without clearing metadata.
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already allocated."
        assert block_id in self.free_block_ids, f"Block {block_id} is not available for activation."
        block.ref_count = 1
        self._move_free_to_used(block_id)
        return block

    def _reference_cached_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"Block {block_id} is not allocated."
        assert block_id in self.used_block_ids, f"Block {block_id} is not in used block set."
        block.ref_count += 1
        return block
    
    def _deallocate_block(self, block_id: int):
        """
        Deallocate a block with given block id. And manage the free and used block ids.
        Args:
            block_id: The id of block to deallocate.
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} cannot be deallocated because it is referenced by {block.ref_count} seqeuences."
        if block.hash == -1:
            block.clear_metadata()
        self.free_block_ids.append(block_id)
        self.used_block_ids.remove(block_id)

    def _find_cached_block_id(self, h: int, token_ids: list[int]) -> int:
        if h == -1:
            return -1

        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1:
            return -1

        block = self.blocks[block_id]
        if block.hash != h:
            if self.hash_to_block_id.get(h) == block_id:
                del self.hash_to_block_id[h]
            return -1
        if block.token_ids != token_ids:
            return -1
        return block_id

    def _max_cacheable_blocks(self, seq: Sequence) -> int:
        # Keep the block containing the final prompt token uncached so prefill
        # still produces logits for sampling the first completion token.
        return max((len(seq)-1)//self.block_size, 0)

    def _build_allocation_plan(self, seq: Sequence):
        h = -1
        max_cacheable_blocks = self._max_cacheable_blocks(seq)
        allocation_plan = []

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            cached_block_id = -1
            if i < max_cacheable_blocks:
                cached_block_id = self._find_cached_block_id(h, token_ids)
            allocation_plan.append((token_ids, h, cached_block_id))

        return allocation_plan

    def _num_free_blocks_needed(self, allocation_plan) -> int:
        num_free_blocks_needed = 0
        counted_cached_blocks = set()
        for _, _, cached_block_id in allocation_plan:
            if cached_block_id == -1:
                num_free_blocks_needed += 1
            elif cached_block_id not in self.used_block_ids and cached_block_id not in counted_cached_blocks:
                num_free_blocks_needed += 1
                counted_cached_blocks.add(cached_block_id)
        return num_free_blocks_needed

    def _allocate_non_reserved_block(self, reserved_block_ids: set[int]) -> Block:
        for block_id in list(self.free_block_ids):
            if block_id not in reserved_block_ids:
                return self._allocate_block(block_id)
        raise RuntimeError("No free KV cache block is available for allocation.")
    
    def can_allocate(self, seq: Sequence) -> bool:
        """
        Check whether the block manager can allocate `num_blocks` blocks for the sequence.
        """
        if seq.num_blocks > self.num_blocks:
            return False
        allocation_plan = self._build_allocation_plan(seq)
        return self._num_free_blocks_needed(allocation_plan) <= len(self.free_block_ids)
    
    def allocate(self, seq: Sequence):
        allocation_plan = self._build_allocation_plan(seq)
        reserved_cached_blocks = {
            cached_block_id
            for _, _, cached_block_id in allocation_plan
            if cached_block_id != -1 and cached_block_id not in self.used_block_ids
        }

        # The sequence needs `num_blocks` blocks to store all tokens,
        # so the block manager needs to allocate `num_blocks` blocks for the sequence.
        for token_ids, h, cached_block_id in allocation_plan:
            if cached_block_id != -1:
                # Update sequence information.
                seq.num_cached_tokens += self.block_size
                
                # Update block information.
                if cached_block_id in self.used_block_ids:
                    block = self._reference_cached_block(cached_block_id)
                # The block used to store this segment of token ids, but now it is not allocated.
                else:
                    block = self._activate_cached_block(cached_block_id)
                    reserved_cached_blocks.discard(cached_block_id)
            else:
                block = self._allocate_non_reserved_block(reserved_cached_blocks)
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
            
