import math
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from babyvllm.config import Config
from babyvllm.models.qwen3 import Qwen3ForCausalLM
from babyvllm.layers.sampler import Sampler
from babyvllm.engine.sequence import Sequence
from babyvllm.utils.context import set_context, reset_context, get_context
from babyvllm.utils.loader import load_model


class ModelRunner:
    """
    ModelRunner acts as a bridge between sequences and model execution.
    It is responsible for:
    (1) Data preparation: `prepare_prefill`, `prepare_decode`, `prepare_sample`.
    (2) Memory management: `warmup_model`, `allocate_kv_cache`.
    (3) Model execution: `capture_cudagraph`, `run_model`, `run`.
    (4) Shared memory communication: `read_shm`, `write_shm`.
    """
    
    def __init__(self, config: Config, rank: int, event: Event|list[Event]):
        self.config = config
        # Event is used to synchronize multi-processes.
        # For rank 0, `event` is a list of events, which size is `world_size-1`.
        # For rank i > 0, `event` is a single event.
        self.event = event
        
        # Set parameters for distributed inference.
        self.world_size = config.tensor_parallel_size
        self.block_size = config.kvcache_block_size
        # Whether to enforce eager execution when running model.
        self.enforce_eager = config.enforce_eager
        
        # Initialize distributed process group.
        self.rank = rank
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', world_size=config.tensor_parallel_size, rank=rank)
        torch.cuda.set_device(rank)
        torch.set_default_device(f'cuda:{rank}')

        self.default_dtype = config.hf_config.dtype
        torch.set_default_dtype(self.default_dtype)

        # Set by prepare_forward() and consumed by run() to avoid re-computing is_decode_only.
        self._is_decode_only = False
        
        # Create model and sampler.
        self.model = Qwen3ForCausalLM(config.hf_config).cuda(rank)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # Get peak memory usage, which is helpful for kv cache allocation.
        self.warmup_model()
        # Allocate kv cache.
        self.allocate_kv_cache()
        
        # Capture CUDA graph for decoding.
        # `self.graphs`: {batch_size : CUDAGraph}
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # Setup shared memory for communication between model runners. (multi-process communication)
        # Rank 0 create the shared memory and child processes link to it.
        # To avoid collision, should be done after all processes finishing model initialization, warmup and kv cache allocation.
        if self.world_size > 1:
            # Synchronize before setting up.
            dist.barrier()
            if self.rank == 0:
                # Clean up existing shared memory.
                try:
                    # `name` is the unique identifier for shared memory.
                    old_shm = SharedMemory(name='babyvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                # Create new shared memory.
                self.shm = SharedMemory(name='babyvllm', create=True, size=2**20)
                # Ensure rank 1 accesses shared memory after rank 0 create it.
                dist.barrier()
            else:
                # Wait until rank 0 create shared memory.
                dist.barrier()
                # Child processes link to the shared memory created by rank 0.
                # (No parameter `create=True` means link to existing shared memory, but not create it.)
                self.shm = SharedMemory(name='babyvllm')
                # Do not call `loop()` in child processes' `__init__`, or it will stuck in an infinite loop.
    
    """
    When LLM engine call a method in rank 0, there is the following steps:
    (1) Rank 0 writes method name and args into shared memory, while other processes wait until events are triggered.
    (2) Rank 0 triggers events to notify other processes.
    (4) Other processes read method name and args from shared memory, and then reset event to un-triggered state.
    (5) All processes call the method.
    """
    
    def write_shm(self, method_name: str, args: tuple):
        """ Write data to shared memory. Only use write when rank == 0. """
        
        assert self.world_size > 1 and self.rank == 0, "Only rank 0 can write shared memory."
        
        # Flatten. For example, if args is (a, b, c), then (method_name, args) is (method_name, (a, b, c)),
        # and (method_name, *args) is (method_name, a, b, c).
        # `pickle.dumps` converts Python object into binary data.
        data = pickle.dumps((method_name, *args))
        n = len(data)
        # Data structure in shared memory:
        # First 4 bytes store the length of data.
        # Next `n` bytes store the pickled data.
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        
        # Trigger events to notify other processes.
        for event in self.event:
            event.set()
    
    def read_shm(self):
        """ Read data from shared memory. Only use read when rank != 0. """
        
        assert self.world_size > 1 and self.rank != 0, "Only rank != 0 can read shared memory."
        
        # Wait until rank 0 write data to shared memory.
        self.event.wait()
        
        n = int.from_bytes(self.shm.buf[:4], 'little')
        # `pickle.loads` converts the binary data into Python object.
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        
        # Reset event to un-triggered state.
        self.event.clear()
        
        return method_name, args
    
    def call(self, method_name: str, *args: dict):
        """ Call a method of the model. It will be used by both rank == 0 and rank != 0. """
        
        # (1) Rank 0 writes method name and args into shared memory.
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, args)
        # (2) Both rank 0 and rank != 0 find the method and call it.
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def exit(self):
        # Close shared memory.
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        
        # Delete CUDA graphs.
        if not self.enforce_eager:
            del self.graphs, self.graph_vars, self.graph_pool
        torch.cuda.synchronize()
        
        # Destroy process group.
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def loop(self):
        """ Rank != 0 loop to read shared memory, wait for event, and call methods. """
        
        assert self.world_size > 1 and self.rank != 0
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == 'exit':
                # Do not need second `exit()` call.
                # self.exit()
                break
    
    @torch.inference_mode()
    def warmup_model(self):
        """ Warmup the model and record the peak memory usage. """
        
        # Cleanup memory pool of PyTorch. It will free unused memory.
        torch.cuda.empty_cache()
        # Reset peak memory usage stats.
        torch.cuda.reset_peak_memory_stats()
        
        # Construct test data: Fill the engine to its maximum token capacity in a single batch.
        num_tokens = self.config.max_num_batched_tokens
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=f'cuda:{self.rank}')
        positions = torch.arange(
            num_tokens, dtype=torch.int64, device=f'cuda:{self.rank}'
        )%self.config.max_model_length
        
        # Bypass the complex logic of `prepare_forward` and manually inject the Context.
        # Due to the lack of allocation of `slot_mapping` and `block_tables`,
        # the attention will enter cache free mode
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32, device=f'cuda:{self.rank}'),
            cu_seqlens_k=torch.tensor([0, num_tokens], dtype=torch.int32, device=f'cuda:{self.rank}'),
            max_seqlen_q=num_tokens,
            max_seqlen_k=num_tokens,
            slot_mapping=None,
            block_tables=None,
            context_lens=None,
        )
        
        # Directly call the model for forward pass.
        self.model(input_ids, positions)
        
        reset_context()
        torch.cuda.empty_cache()
    
    def allocate_kv_cache(self):
        # ===== (1) Find available memory for kv cache. =====
        # `total_mem` is the total memory of the current device.
        # `total_mem*self.config.gpu_memory_utilization` is the memory allowed to use, reserving some for other purposes, such as cuda context.
        free_mem, total_mem = torch.cuda.mem_get_info()
        # `used_mem` is the memory already used, which is cannot be allocated to kv cache.
        used_mem = total_mem-free_mem
        # `peak_mem_usage` is the maximum allocated memory begin from `reset_peak_memory_stats()` in model warmup,
        # which is the peak memory usage of the model execution.
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        # `current_mem_usage` is the current allocated memory. Because `empty_cache()` is called at the end of model warmup,
        # current memory usage is much smaller than peak memory usage.
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # `peak_mem_usage-current_mem_usage` is the possible memory usage increase during model execution,
        # so it should be reserved to ensure the model can run without OOM.
        available_mem = total_mem*self.config.gpu_memory_utilization-used_mem-peak_mem_usage+current_mem_usage
        
        # ===== (2) Compute kv cache block size and number of available blocks. =====
        num_layers = self.config.hf_config.num_hidden_layers
        num_kv_heads = self.config.hf_config.num_key_value_heads//self.world_size
        head_dim = self.config.hf_config.head_dim
        # size of one kv cache block in bytes
        # "*2" because we need to store both key and value.
        block_size_bytes = self.block_size*2*num_layers*num_kv_heads*head_dim*self.default_dtype.itemsize
        self.config.num_kvcache_blocks = int(available_mem)//block_size_bytes
        assert self.config.num_kvcache_blocks >= 1, f"Not enough memory to hold even one kv cache block on rank {self.rank}."
    
        # ===== (3) Allocate memory for kv cache. =====
        # Although `allocated_kv_cache` is a local variable, it will not be deleted out of the function,
        # because it will be referred by kv cache variables in model layers.
        allocated_kv_cache = torch.empty(2, num_layers, self.config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')

        # ===== (4) Divide the giant kv cache pool into blocks and assign blocks to layers in model. =====
        layer_id = 0
        for module in self.model.modules():
            # `Attention` layer has `k_cache` and `v_cache` attributes.
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        max_bs = self.config.max_num_sequences
        max_len = self.config.max_model_length
        max_num_blocks = math.ceil(max_len/self.block_size)
        
        # Create fake inputs for capturing CUDA graph with maximum batch size and maximum sequence length.
        # In decode phase, input is a single token id for each sequence, so the shape is always (batch_size,).
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=f'cuda:{self.rank}')
        positions = torch.zeros(max_bs, dtype=torch.int64, device=f'cuda:{self.rank}')
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=f'cuda:{self.rank}')
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=f'cuda:{self.rank}')
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        outputs = torch.zeros(max_bs, self.config.hf_config.hidden_size, device=f'cuda:{self.rank}')
        
        # Which batch sizes we want to capture CUDA graph for.
        self.graph_batch_sizes = [1, 2, 4, 8]+list(range(16,max_bs+1, 16))
        # {batch_size : CUDAGraph}
        self.graphs = {}
        # Graph pool allows to reuse memory of CUDA graph with different batch sizes.
        self.graph_pool = None
        
        for batch_size in reversed(self.graph_batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                block_tables=block_tables[:batch_size],
                context_lens=context_lens[:batch_size],
                num_decode_seqs=batch_size,
                num_decode_tokens=batch_size,
                prefill_max_seqlen_q=0,
                prefill_max_seqlen_k=0,
                cu_seqlens_q_offset=0,
                cu_seqlens_k_offset=0,
            )
            # Warm up.
            # Complete memory allocation before capturing CUDA graph,
            # to ensure stable memory allocation during capturing.
            outputs[:batch_size] = self.model(input_ids[:batch_size], positions[:batch_size])
            
            # Capture CUDA graph.
            # In the context of `torch.cuda.graph`, PyTorch will record:
            # (1) All CUDA kernel calls and their parameters.
            # (2) All memory accesses, including addresses of
            #       input tensor, model parameters, output tensor, and temporary tensors during the forward pass.
            # These information will be saved in torch.cuda.CUDAGraph object.
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size], positions[:batch_size])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[batch_size] = graph
            
            # Make sure the capture is done before moving to the next capture.
            torch.cuda.synchronize()
            reset_context()
        
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
    
    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare sampling temperature for each sequence. """
        
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    def prepare_forward(self, seqs: list[Sequence]) -> torch.Tensor:
        """
        Unified data preparation for both prefill, chunked prefill, decode, and mixed batches.
        """

        # Sort sequences so Decode is before Prefill.
        # Sorting rules:
        #   (1) Decode (q_len=1) comes first.  Decode => sort key=0, Prefill => sort key=1
        #   (2) Within Prefill group, larger chunk_size comes first to reduce KV cache fragmentation.
        #
        # How to judge if a sequence is Decode:
        # (num_computed_tokens>0) and (num_computed_tokens==num_tokens-1)
        #
        # For example, if the batch has 5 sequences:
        #   Seq A: num_computed=100, num_tokens=101, chunk=1   -> decode,   key=(0, -1)
        #   Seq B: num_computed=200, num_tokens=201, chunk=1   -> decode,   key=(0, -1)
        #   Seq C: num_computed=50,  num_tokens=256, chunk=100 -> prefill,  key=(1, -100)
        #   Seq D: num_computed=0,   num_tokens=256, chunk=256 -> prefill,  key=(1, -256)
        #   Seq E: num_computed=30,  num_tokens=128, chunk=50  -> prefill,  key=(1, -50)
        # Sorted order: [A, B, D, C, E]
        #   Decode: A, B (num_decode_seqs=2, num_decode_tokens=2)
        #   Prefill: D(256), C(100), E(50)
        #
        # sort_permutation[i]: 
        #   The original index of the i-th sequence in the sorted batch.
        #   Used in `run()` to recover the original order of the sampled token_ids.
        #   For example: 
        #   [A, B, C, D, E] -> Sorted order: [A, B, D, C, E] -> sort_permutation = [0, 1, 3, 2, 4]
        indexed_seqs = sorted(enumerate(seqs), key=lambda pair: (
            0 if (pair[1].num_computed_tokens > 0 and pair[1].num_computed_tokens == pair[1].num_tokens - 1) else 1,
            -(pair[1].chunk_size if hasattr(pair[1], 'chunk_size') else 0)
        ))
        self._sort_permutation = [idx for idx, _ in indexed_seqs]  # sort_permutation[i] = original_index
        seqs = [seq for _, seq in indexed_seqs]

        # All tokens and their positions that will be passed to the model for forward pass.
        input_ids = []
        positions = []

        # Assume there are three sequences in the current batch:
        # Seq0: Prefill phase. Just start computing. chunk_size = 256.
        # Seq1: Chunked-prefill phase. Already computed 100 tokens, now computing 50 tokens. chunk_size = 50.
        # Seq2: Decode phase. Already computed 1000 tokens, now computing 1 token. chunk_size = 1.
        # Number of tokens in each sequence that will participate in attention,
        # which is the number of `query` of each sequence and equals to `chunk_size`.
        # [256, 50, 1]
        seqlens_q = []
        # Prefix sum of `seqlens_q`.
        # [0, 256, 306, 307]
        cu_seqlens_q = [0]
        # Number of previous tokens in each sequence that each `query` will attend to,
        # which is the number of `key` of each sequence and equals to `already_computed_tokens+chunk_size`.
        # [256, 150, 1001]
        seqlens_k = []
        # Prefix sum of `seqlens_k`.
        # [0, 256, 406, 1407]
        cu_seqlens_k = [0]

        # Number of tokens that have been stored in kv cache for each sequence.
        # [256, 150, 1001]
        context_lens = []

        # The two-dimensional matrix after padding the `seq.block_table` of each sequence.
        # When calculating attention, when the operator needs to find the Nth historical token,
        # it will use `block_tables` to check the physical memory.
        block_tables = []
        # The specific location of physical memory (`allocated_kv_cache`)
        # where the KV of tokens in `input_ids` should be written to.
        # This is an array of the same length as `input_ids`.
        # In Triton operator `store_kvcache`, the GPU will store the calculated KV
        # in the slot indicated by `slot_mapping` (physical block number*block size+offset within block).
        slot_mapping = []

        for seq in seqs:
            start_idx = seq.num_computed_tokens
            chunk_size = seq.chunk_size
            end_idx = start_idx+chunk_size
            
            # 1. Get tokens and their positions that will participate in attention.
            process_tokens = seq.token_ids[start_idx:end_idx]
            input_ids.extend(process_tokens)
            positions.extend(list(range(start_idx, end_idx)))
            
            # 2. Length information in attention, which is important for FlashAttention.
            q_len = chunk_size
            k_len = end_idx
            
            seqlens_q.append(q_len)
            seqlens_k.append(k_len)
            cu_seqlens_q.append(cu_seqlens_q[-1]+q_len)
            cu_seqlens_k.append(cu_seqlens_k[-1]+k_len)
            context_lens.append(k_len)
            
            # 3. Which physical memory location should be used to store the tokens.
            for logical_idx in range(start_idx, end_idx):
                block_idx = logical_idx//self.block_size
                block_offset = logical_idx%self.block_size
                physical_block_id = seq.block_table[block_idx]
                slot_mapping.append(physical_block_id*self.block_size+block_offset)
        
        # Fill in `block_tables` to pass in Triton Kernel and FlashAttention to access historical KV.    
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max((len(bt) for bt in all_block_tables), default=0)
        if max_num_blocks > 0:
            for seq in seqs:
                aligned_block_table = seq.block_table+[-1]*(max_num_blocks-len(seq.block_table))
                block_tables.append(aligned_block_table)
            block_tables_tensor = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            block_tables_tensor = None
        
        
        # Transfer data to GPU.
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        
        # Only when all sequences are in the decode phase, this batch can be counted as `is_decode_only`.
        # Pure decode batches can use CUDA Graph replay (10x faster than eager execution).
        # Save on the instance so run() can reuse it without a second O(N) scan.
        self._is_decode_only = all(seq.num_computed_tokens > 0 and seq.num_computed_tokens == seq.num_tokens-1 for seq in seqs)

        # Calculate split point for mixed batch reordering.
        # After sorting, Decode sequences are first, Prefill sequences are after.
        # Count how many are Decode sequences — this is our "split point" between the two groups.
        # Since each Decode sequence contributes exactly 1 token,
        # num_decode_tokens == num_decode_seqs.
        # Example: after reorder, seqs = [D0, D1, D2, P0, P1]
        #   num_decode_seqs = 3  (The first 3 sequences are Decode sequences.)
        #   num_decode_tokens = 3  (The first 3 positions in q tensor are Decode tokens.)
        #   Prefill tokens start from q[3].
        num_decode_seqs = sum(1 for s in seqs if s.num_computed_tokens > 0 and s.num_computed_tokens == s.num_tokens - 1)
        num_decode_tokens = num_decode_seqs

        # Pre-compute Prefill sub-batch scalars on the CPU side to avoid
        # GPU-CPU sync (.item()) in Attention.forward().
        if num_decode_seqs < len(seqs):
            prefill_max_seqlen_q = max(seqlens_q[num_decode_seqs:])
            prefill_max_seqlen_k = max(seqlens_k[num_decode_seqs:])
            cu_seqlens_q_offset = cu_seqlens_q[num_decode_seqs]
            cu_seqlens_k_offset = cu_seqlens_k[num_decode_seqs]
        else:
            prefill_max_seqlen_q = 0
            prefill_max_seqlen_k = 0
            cu_seqlens_q_offset = 0
            cu_seqlens_k_offset = 0

        set_context(
            is_prefill=not self._is_decode_only,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q, default=0),
            max_seqlen_k=max(seqlens_k, default=0),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_tables=block_tables_tensor,
            context_lens=torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            num_decode_seqs=num_decode_seqs,
            num_decode_tokens=num_decode_tokens,
            prefill_max_seqlen_q=prefill_max_seqlen_q,
            prefill_max_seqlen_k=prefill_max_seqlen_k,
            cu_seqlens_q_offset=cu_seqlens_q_offset,
            cu_seqlens_k_offset=cu_seqlens_k_offset,
        )
        
        return input_ids, positions
    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_decode_only: bool):
        """
        Run model inference for a batch of sequences and return logits.
        
        Args:
            input_ids: Input token ids. shape: (total_tokens,)
            positions: Positions of each token. shape: (total_tokens,)
            is_decode_only: Whether there is only decode requests in the batch.
        """
        
        # Mixed batch has highly dynamic variable tensor shape,
        # but CUDA graph can not capture variable tensor shape.
        # So, only when the batch is in pure decode phase and eager mode is not used, we can use CUDA graph.
        if not is_decode_only or self.enforce_eager:
            hidden_states = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # Find the CUDA graph that can handle current batch size while minimize memory waste.
            # `next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)`
            # finds the smallest batch size that is >= to current batch size.
            graph = self.graphs[next(bs_ for bs_ in self.graph_batch_sizes if bs_ >= bs)]

            # Copy input data into graph variables.
            # Do not change memory layout which has been captured. Use in-place operations.
            graph_vars = self.graph_vars
            graph_vars['input_ids'][:bs] = input_ids
            graph_vars['positions'][:bs] = positions
            graph_vars['slot_mapping'].fill_(-1)
            graph_vars['slot_mapping'][:bs] = context.slot_mapping
            graph_vars['context_lens'].zero_()
            graph_vars['context_lens'][:bs] = context.context_lens
            graph_vars['block_tables'][:bs, :context.block_tables.size(1)] = context.block_tables

            # Replay CUDA graph.
            graph.replay()
            logits = self.model.compute_logits(graph_vars['outputs'][:bs])

        return logits
    
    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
        """ Run model inference for a batch of sequences and return output token ids. """

        # Prepare sampling temperatures in original order before prepare_forward
        # reorders sequences. They will be reordered below to match sorted logits.
        temperatures_original = self.prepare_sample(seqs) if self.rank == 0 else None

        # Prepare the data for forward pass (prefill or decode).
        # prepare_forward() sorts seqs internally (Decode first, Prefill after)
        # and records the permutation in self._sort_permutation.
        input_ids, positions = self.prepare_forward(seqs)

        # Reorder temperatures to match the sorted logits order.
        if self.rank == 0:
            temperatures = temperatures_original[self._sort_permutation]
        else:
            temperatures = None

        # Execute model inference.
        # is_decode_only was already computed in prepare_forward() and saved on the instance.
        logits = self.run_model(input_ids, positions, self._is_decode_only)

        # Sample tokens from logits.
        # Convert token ids to list of int, since sequence only supports int token ids.
        if self.rank == 0:
            # token_ids_sorted has len(seqs) entries — one sampled token per sequence
            # in sorted order (Decode first, Prefill after).
            token_ids_sorted = self.sampler(logits, temperatures).tolist()
            # Unsort back to original sequence order.
            # _sort_permutation[i] = original_index of the i-th sorted sequence.
            token_ids = [0] * len(seqs)
            for sorted_pos, original_pos in enumerate(self._sort_permutation):
                token_ids[original_pos] = token_ids_sorted[sorted_pos]
        else:
            token_ids = None

        # Reset context for next forward pass.
        reset_context()

        return token_ids