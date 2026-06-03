from collections import deque
from dataclasses import dataclass

from babyvllm.config import Config
from babyvllm.engine.block_manager import BlockManager
from babyvllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class ScheduledBatch:
    """Logical scheduler result split into physical Decode/Prefill sub-batches."""

    decode_sequences: list[Sequence]
    prefill_sequences: list[Sequence]

    @property
    def all_sequences(self) -> list[Sequence]:
        return self.decode_sequences+self.prefill_sequences

    def __bool__(self) -> bool:
        return bool(self.decode_sequences or self.prefill_sequences)


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

        # Logical scheduling can select Decode and Prefill work in the same
        # round. The engine/model runner executes them as separate physical
        # forwards so Decode keeps CUDA Graph replay while Prefill remains eager.
        self.max_prefill_tokens_per_step = config.max_prefill_tokens_per_step
        self.max_prefill_chunk_size = config.max_prefill_chunk_size

        self.stats = {
            "pure_decode": 0,
            "pure_prefill": 0,
            "mixed": 0,
            "preempt": 0,
        }
        
    def is_finished(self):
        """ Check if all sequences have finished. """
        
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)

    def get_stats(self) -> dict[str, int]:
        """Return scheduler instrumentation counters."""

        return dict(self.stats)

    def _is_decode_seq(self, seq: Sequence) -> bool:
        return seq.num_computed_tokens > 0 and seq.num_computed_tokens == seq.num_tokens - 1

    def _has_decode_seq(self) -> bool:
        return any(self._is_decode_seq(seq) for seq in self.running)

    def _has_pending_prefill(self) -> bool:
        return bool(self.waiting) or any(not self._is_decode_seq(seq) for seq in self.running)

    def _get_decode_block_reserve(self) -> int:
        """Reserve one free KV block per active Decode sequence for future growth."""

        return sum(1 for seq in self.running if self._is_decode_seq(seq))

    def _record_batch_stats(self, batch: ScheduledBatch):
        if not batch:
            return
        has_decode = bool(batch.decode_sequences)
        has_prefill = bool(batch.prefill_sequences)
        if has_decode and has_prefill:
            self.stats["mixed"] += 1
        elif has_decode:
            self.stats["pure_decode"] += 1
        else:
            self.stats["pure_prefill"] += 1

    def preempt(self, seq: Sequence):
        """ Preempt a sequence. Deallocate its cached blocks and put it back to waiting queue. """

        self.stats["preempt"] += 1
        self.block_manager.deallocate(seq)
        # Reset the computation progress of the preempted sequence.
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
    
    def _get_max_chunk_size(
        self,
        seq: Sequence,
        remaining_budget: int,
        reserve_blocks: int = 0,
        max_chunk_size: int | None = None,
    ) -> int:
        """ Get the maximum token budget that can be allocated to this sequence. """
        
        # Limit 1: Cannot allocate more tokens than uncomputed tokens.
        uncomputed = seq.num_tokens-seq.num_computed_tokens
        # Limit 2: Cannot allocate more tokens than remaining budget.
        chunk_size = min(uncomputed, remaining_budget)
        if max_chunk_size is not None:
            chunk_size = min(chunk_size, max_chunk_size)
        
        # Limit 3: Cannot allocate more tokens than remaining physical memory.
        free_blocks = max(len(self.block_manager.free_block_ids)-reserve_blocks, 0)
        if seq.num_computed_tokens%self.block_manager.block_size == 0:
            max_tokens_by_blocks = free_blocks*self.block_manager.block_size
        else:
            space_in_last_block = self.block_manager.block_size-seq.num_computed_tokens%self.block_manager.block_size
            max_tokens_by_blocks = space_in_last_block+free_blocks*self.block_manager.block_size

        return min(chunk_size, max_tokens_by_blocks)

    def _schedule_decode_batch(
        self,
        max_sequences: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Sequence]:
        """Schedule a pure Decode batch from running sequences."""

        max_sequences = self.max_num_sequences if max_sequences is None else max_sequences
        max_tokens = self.max_num_batched_tokens if max_tokens is None else max_tokens
        scheduled_sequences = []
        current_scheduled_tokens = 0
        num_running = len(self.running)

        for _ in range(num_running):
            if not self.running:
                break
            if len(scheduled_sequences) >= max_sequences:
                break
            if current_scheduled_tokens >= max_tokens:
                break

            seq: Sequence = self.running.popleft()
            if not self._is_decode_seq(seq):
                self.running.append(seq)
                continue

            chunk_size = 1
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

        if scheduled_sequences:
            self.running.extend(scheduled_sequences)

        return scheduled_sequences

    def _schedule_prefill_batch(
        self,
        max_sequences: int | None = None,
        max_tokens: int | None = None,
        reserve_blocks: int = 0,
    ) -> list[Sequence]:
        """Schedule a pure Prefill batch from running chunks and waiting requests."""

        max_sequences = self.max_num_sequences if max_sequences is None else max_sequences
        max_tokens = self.max_num_batched_tokens if max_tokens is None else max_tokens
        max_tokens = min(max_tokens, self.max_prefill_tokens_per_step)
        scheduled_sequences = []
        current_scheduled_tokens = 0

        # Continue already-admitted chunked prefills first. They already own KV
        # blocks, so finishing them avoids leaving half-prefilled work resident.
        num_running = len(self.running)
        for _ in range(num_running):
            if len(scheduled_sequences) >= max_sequences:
                break
            if current_scheduled_tokens >= max_tokens:
                break

            seq: Sequence = self.running.popleft()
            if self._is_decode_seq(seq):
                self.running.append(seq)
                continue

            remaining_budget = max_tokens-current_scheduled_tokens
            chunk_size = self._get_max_chunk_size(
                seq,
                remaining_budget,
                reserve_blocks,
                max_chunk_size=self.max_prefill_chunk_size,
            )

            if chunk_size > 0:
                self.block_manager.allocate_chunk(seq, chunk_size)
                seq.chunk_size = chunk_size
                scheduled_sequences.append(seq)
                current_scheduled_tokens += chunk_size

            self.running.append(seq)

        # Admit new requests only into a pure Prefill window. When Decode
        # sequences are resident, keep a small KV reserve so future Decode block
        # extensions do not immediately force preemption/recompute.
        while self.waiting and len(scheduled_sequences) < max_sequences:
            if current_scheduled_tokens >= max_tokens:
                break

            seq: Sequence = self.waiting[0]
            remaining_budget = max_tokens-current_scheduled_tokens
            chunk_size = self._get_max_chunk_size(
                seq,
                remaining_budget,
                reserve_blocks,
                max_chunk_size=self.max_prefill_chunk_size,
            )

            if chunk_size > 0:
                self.block_manager.allocate_chunk(seq, chunk_size)
                seq.status = SequenceStatus.RUNNING
                seq.chunk_size = chunk_size

                self.waiting.popleft()
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += chunk_size
            else:
                # FCFS admission: do not skip the head request just because a
                # later shorter prompt might fit.
                break

        return scheduled_sequences
    
    def schedule(self) -> ScheduledBatch:
        """
        Schedule sequences. Allocate resources for scheduled sequences.
        Support continuous batching at admission time, while keeping each
        physical forward as pure Decode or pure Prefill whenever possible.
        Support chunked prefilling.
        
        Why use `seq.num_tokens` instead of `seq.num_prompt_tokens` in many judgements?
        Suppose the initial prompt length sent by the user is 100.
        [Phase 1: Normal Chunked Prefill (e.g., chunk by 60)]
        - Step 1: num_computed_tokens=0,  num_tokens=100 -> need 100 more -> compute 60 tokens
        - Step 2: num_computed_tokens=60, num_tokens=100 -> need 40 more  -> compute 40 tokens, then output token 101 and append
        [Phase 2: Normal Decode]
        - Step 3: num_computed_tokens=100, num_tokens=101 -> need 1 more   -> Decode operator, output token 102 and append
        - Step 4: num_computed_tokens=101, num_tokens=102 -> need 1 more   -> Decode operator, output token 103 and append
        [Phase 3: Critical Preemption Recovery]
        - Assume after computing token 103, it gets preempted due to memory pressure.
        - During recovery, its KV cache is cleared and num_computed_tokens is reset to 0.
        - Current state: num_computed_tokens=0, num_tokens=103.
        - If using num_prompt_tokens(100): need 100-0=100. Only first 100 tokens are recovered, last 3 tokens lost forever, system crashes.
        - If using num_tokens(103): need 103-0=103. System treats generated 3 tokens as prompt and also recovers them.
        """
        
        decode_sequences = self._schedule_decode_batch()

        remaining_sequences = self.max_num_sequences-len(decode_sequences)
        remaining_tokens = self.max_num_batched_tokens-len(decode_sequences)
        prefill_sequences = []

        if self._has_pending_prefill() and remaining_sequences > 0 and remaining_tokens > 0:
            reserve_blocks = self._get_decode_block_reserve() if decode_sequences else 0
            prefill_sequences = self._schedule_prefill_batch(
                max_sequences=remaining_sequences,
                max_tokens=remaining_tokens,
                reserve_blocks=reserve_blocks,
            )

        batch = ScheduledBatch(
            decode_sequences=decode_sequences,
            prefill_sequences=prefill_sequences,
        )
        self._record_batch_stats(batch)
        return batch
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, list[int], bool]]:
        """
        Append generated tokens to sequences and handle finished sequences.
        When chunked prefill is enabled, if a prompt is cut into 4 pieces,
        the token IDs returned by the model during the first three operation are meaningless,
        because the context is not yet complete.
        We must discard them and only accept them when the last piece is calculated.
        """
        
        step_outputs = []
        for seq, token_id in zip(seqs, token_ids):
            # ======================
            # (1) Accumulate the number of handled chunks.
            # ======================
            chunk_size = getattr(seq, "chunk_size", 1)
            seq.num_computed_tokens += chunk_size
            
            # ======================
            # (2) Append the token to the sequence and check if the sequence is finished.
            # (only if the prompt has been computed already).
            # ======================
            if seq.num_computed_tokens >= seq.num_tokens:
                # The sequence has finished prefilling phase or it is already in decoding phase.

                # Check max_tokens BEFORE appending so that max_tokens=0
                # finishes immediately without generating any completion tokens.
                if seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    step_outputs.append((seq.seq_id, [], True))
                    continue

                seq.append_token(token_id)

                # In following cases, a sequence will stop append new tokens:
                # (a) The eos token is generated.
                stop_check_eos = not seq.ignore_eos and token_id == self.eos
                # (b) The number of completion tokens exceeds the limit.
                stop_check_max_completion = seq.num_completion_tokens >= seq.max_tokens
                # (c) The sequence length exceeds the maximum model length.
                stop_check_max_model_len = seq.max_model_length is not None and len(seq) >= seq.max_model_length

                if stop_check_eos or stop_check_max_completion or stop_check_max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                step_outputs.append((seq.seq_id, [token_id], seq.is_finished))
            else:
                # The sequence is still in chunked prefilling phase.
                # Ignore its output token.
                pass
        return step_outputs

    def abort_sequence(self, seq_id: int) -> bool:
        """Abort a sequence by seq_id. Free KV cache blocks and remove from deques."""
        for i, seq in enumerate(self.waiting):
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                del self.waiting[i]
                return True
        for i, seq in enumerate(self.running):
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                del self.running[i]
                return True
        return False
