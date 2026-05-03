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
    
    def _get_max_chunk_size(self, seq: Sequence, remaining_budget: int) -> int:
        """ Get the maximum token budget that can be allocated to this sequence. """
        
        # Limit 1: Cannot allocate more tokens than uncomputed tokens.
        # Limit 2: Cannot allocate more tokens than remaining budget.
        uncomputed = seq.num_prompt_tokens-seq.num_computed_tokens
        chunk_size = min(uncomputed, remaining_budget)
        
        # Limit 3: Cannot allocate more tokens than remaining physical memory.
        free_blocks = len(self.block_manager.free_block_ids)
        if seq.num_computed_tokens%self.block_manager.block_size == 0:
            max_tokens_by_blocks = free_blocks*self.block_manager.block_size
        else:
            space_in_last_block = self.block_manager.block_size-seq.num_computed_tokens%self.block_manager.block_size
            max_tokens_by_blocks = space_in_last_block+free_blocks*self.block_manager.block_size

        return min(chunk_size, max_tokens_by_blocks)
    
    def schedule(self) -> list[Sequence]:
        """
        Schedule sequences. Allocate resources for scheduled sequences. 
        Support continuous batching. Mix prefilling and decoding sequences in one scheduling.
        Support chunked prefilling.
        """
        
        scheduled_sequences = []
        current_scheduled_tokens = 0
        scheduled_decodes_ids = []
        scheduled_prefill_ids = []
        
        # ======================
        # (1) Firstly, try schedule decoding sequences from running queue.
        # ======================       
        # Each scheduling of decoding sequence will generate 1 token, thus cost 1 token budget.
        while self.running and len(scheduled_sequences) < self.max_num_sequences:
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                    break
            
            seq: Sequence = self.running.popleft()
            
            if seq.num_computed_tokens < seq.num_prompt_tokens:
                # The sequence is in chunked prefilling phase.
                remaining_budget = self.max_num_batched_tokens-current_scheduled_tokens
                chunk_size = self._get_max_chunk_size(seq, remaining_budget)
                
                if chunk_size > 0:
                    # Physical memory is enough.
                    self.block_manager.allocate_chunk(seq, chunk_size)
                    seq.chunk_size = chunk_size
                    scheduled_sequences.append(seq)
                    current_scheduled_tokens += chunk_size
                    scheduled_prefill_ids.append(seq.seq_id)
                else:
                    # Physical memory is not enough, try to preempt other sequences.
                    if self.running:
                        # If there are other running sequences, try to preempt the last one.
                        self.preempt(self.running.pop())
                        # After preempting, put the sequence back to the head of running queue,
                        # and try to schedule it in the next round.
                        self.running.appendleft(seq)
                    else:
                        # Otherwise, the sequence can not be scheduled.
                        # It can only free its cached blocks and wait for the next scheduling.
                        self.preempt(seq)
                        break
            else:
                # The sequence is in decoding phase.
                chunk_size = 1
                # If there is no free physical memory, try to preempt other sequences.
                # Preempting one running sequence may be not enough,
                # so use `while` instead of `if` like prefilling phase.
                while not self.block_manager.can_allocate_chunk(seq, chunk_size):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    self.block_manager.allocate_chunk(seq, chunk_size)
                    seq.chunk_size = chunk_size
                    scheduled_sequences.append(seq)
                    current_scheduled_tokens += chunk_size
                    scheduled_decodes_ids.append(seq.seq_id)
        
        # We suppose the sequences scheduled in this round is not finished,
        # and put them back to running queue.
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        # ======================
        # (2) Next, try schedule prefilling sequences from waiting queue using remaining token budget.
        # ======================
        remaining_seq_budget = self.max_num_sequences-len(scheduled_sequences)
        scheduled_new_prefills = []
        
        while self.waiting and len(scheduled_new_prefills) < remaining_seq_budget:
            seq: Sequence = self.waiting[0]
            remaining_budget = self.max_num_batched_tokens-current_scheduled_tokens
            
            chunk_size = self._get_max_chunk_size(seq, remaining_budget)
            
            if chunk_size > 0:
                # Physical memory is enough.
                self.block_manager.allocate_chunk(seq, chunk_size)
                seq.status = SequenceStatus.RUNNING
                seq.chunk_size = chunk_size
                
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new_prefills.append(seq)
                current_scheduled_tokens += chunk_size
                scheduled_prefill_ids.append(seq.seq_id)
            else:
                # Physical memory is not enough.
                # Halt scheduling upon encountering the first prefill request,
                # that cannot be accommodated within the remaining budget.
                # It ensures the First Come First Serve (FCFS) order of prefill requests,
                # preventing later short requests from cutting in line.
                break
        
        scheduled_sequences.extend(scheduled_new_prefills)
        
        return scheduled_sequences
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        Append generated tokens to sequences and handle finished sequences.
        When chunked prefill is enabled, if a prompt is cut into 4 pieces,
        the token IDs returned by the model during the first three operation are meaningless,
        because the context is not yet complete.
        We must discard them and only accept them when the last piece is calculated.
        """
        
        for seq, token_id in zip(seqs, token_ids):
            # ======================
            # (1) Accumulate the number of handled chunks.
            # ======================
            chunk_size = getattr(seq, "chunk_size", 1)
            seq.num_computed_tokens += chunk_size
            
            # ======================
            # (2) Append the token to the sequence.
            # ======================
            if seq.num_computed_tokens >= seq.num_prompt_tokens:
                # The sequence has finished prefilling phase or it is already in decoding phase.
                seq.append_token(token_id)
            else:
                # The sequence is still in chunked prefilling phase.
                # Ignore its output token.
                pass
            
            # ======================
            # (3) Check if the sequence is finished (only if the prompt has been computed already).
            # ======================
            if seq.num_computed_tokens >= seq.num_prompt_tokens:
                # In following cases, a sequence will stop append new tokens:
                # (a) The eos token is generated.
                stop_check_eos = not seq.ignore_eos and token_id == self.eos
                # (b) The number of completion tokens exceeds the limit.
                stop_check_max_completion = 1+seq.num_completion_tokens >= seq.max_tokens
                # (c) The sequence length exceeds the maximum model length.
                stop_check_max_model_len = seq.max_model_length is not None and len(seq) >= seq.max_model_length
                
                if stop_check_eos or stop_check_max_completion or stop_check_max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
    