from collections import deque

from babyvllm.config import Config
from babyvllm.engine.block_manager import BlockManager
from babyvllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        """
        (1) Manage kv cache blocks. Create the BlockManager object.
        (2) Track sequence status. Create the waiting queue and running queue.
        (3) Limitations. Limit the number of sequences and tokens.
        
        Args:
            max_num_sequences: The maximum number of sequences that can be scheduled in a single run.
            max_num_batched_tokens: The maximum number of tokens that can be scheduled in a single run.
            max_cached_blocks: The maximum number of cached blocks.
            block_size: The size of each block.
            eos: The end-of-sequence token id.
        """
        
        # block manager
        self.block_manager = BlockManager(num_blocks=config.num_kvcache_blocks, block_size=config.kvcache_block_size)
        self.max_num_sequences = config.max_num_sequences
        self.max_num_batched_tokens = config.max_num_batched_tokens
        
        # sequence queue
        self.waiting = deque()
        self.running = deque()
        self.eos = config.hf_config.eos_token_id if config.hf_config.eos_token_id is not None else config.eos
        
    def is_finished(self):
        """ Check if all sequences have finished. """
        
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)
    
    def preempt(self, seq: Sequence):
        """ Preempt a sequence. Deallocate its cached blocks and put it back to waiting queue. """
        
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
    
    def schedule(self) -> list[Sequence]:
        """
        Schedule sequences. Allocate resources for scheduled sequences. 
        Support continuous batching. Mix prefilling and decoding sequences in one scheduling.
        """
        
        scheduled_sequences = []
        current_scheduled_tokens = 0
        
        # ======================
        # (1) Try schedule decoding sequences from running queue.
        # ======================
        # Each scheduling of decoding sequence will generate 1 token, thus cost 1 token budget.
        while self.running and len(scheduled_sequences) < self.max_num_sequences:
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                    break
            
            seq = self.running.popleft()
            
            # If there is no free cache block, try to preempt other sequences.
            # Preempting one running sequence may be not enough, so use `while` instead of `if`.
            while not self.block_manager.can_append(seq):
                if self.running:
                    # If there are other running sequences, try to preempt the last one.
                    self.preempt(self.running.pop())
                else:
                    # Otherwise, the sequence can not be scheduled.
                    # It can only free its cached blocks and wait for the next scheduling.
                    self.preempt(seq)
                    break
            else:
                # Allocate resources for the sequence.
                self.block_manager.append(seq)
                # Manage the scheduled sequence.
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1
        
        # If successfully scheduled sequences, put them back to running queue in the same order.
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        
        # ======================
        # (2) Try schedule prefilling sequences from waiting queue using remaining token budget.
        # ======================
        remaining_seq_budget = self.max_num_sequences-len(scheduled_sequences)
        scheduled_prefills = []
        
        while self.waiting and len(scheduled_prefills) < remaining_seq_budget:
            seq = self.waiting[0]
            
            # Check if remaining token budget is enough for the sequence,
            # and if the block manager can allocate resources for the sequence.
            if (len(seq)+current_scheduled_tokens <= self.max_num_batched_tokens) and self.block_manager.can_allocate(seq):
                # Allocate resources for the sequence.
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                
                # Manage the scheduled sequence.
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_prefills.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                # Halt scheduling upon encountering the first prefill request,
                # that cannot be accommodated within the remaining budget.
                # It ensures the First Come First Serve order of prefill requests,
                # preventing later short requests from cutting in line.
                break
            
        scheduled_sequences.extend(scheduled_prefills)
        
        return scheduled_sequences
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """ Postprocess scheduled sequences. Append generated tokens to sequences and handle finished sequences. """
        
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # In following cases, a sequence will stop append new tokens:
            # (1) The eos token is generated.
            stop_check_eos = not seq.ignore_eos and token_id == self.eos
            # (2) The number of completion tokens exceeds the limit.
            stop_check_max_completion = 1+seq.num_completion_tokens >= seq.max_tokens
            # (3) The sequence length exceeds the maximum model length.
            stop_check_max_model_len = seq.max_model_length is not None and len(seq) >= seq.max_model_length
            
            if stop_check_eos or stop_check_max_completion or stop_check_max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
    