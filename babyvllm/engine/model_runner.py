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
        
    def warmup_model(self):
        """ Warmup the model and record the peak memory usage. """
        
        # Cleanup memory pool of PyTorch. It will free unused memory.
        torch.cuda.empty_cache()
        # Reset peak memory usage stats.
        torch.cuda.reset_peak_memory_stats()
        
        # Create a batch of fake sequences and run the model.
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_length = self.config.max_model_length
        batch_size = min(max_num_batched_tokens//max_model_length, self.config.max_num_sequences)
        seqs = [Sequence(token_ids=[0]*max_model_length) for _ in range(batch_size)]
        self.run(seqs=seqs, is_prefill=True)
        torch.cuda.empty_cache()
    
    def allocate_kv_cache(self):
        # ===== (1) Find available memory while reserving room for model execution. =====
        # Get free memory and total memory of the current device.
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem*self.config.gpu_memory_utilization
        # Reserve room for model execution.
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        available_mem = total_free_mem-(peak_mem_usage-current_mem_usage)
        
        # ===== (2) Find parameters to compute kv cache size. =====
        num_layers = self.config.hf_config.num_hidden_layers
        num_kv_heads = self.config.hf_config.num_key_value_heads//self.world_size
        head_dim = self.config.hf_config.head_dim
        
        # ===== (3) Compute kv cache block size and number of available blocks. =====
        # size of one kv cache block in bytes
        # "*2" because we need to store both key and value.
        block_size_bytes = self.block_size*2*num_layers*num_kv_heads*head_dim*self.default_dtype.itemsize
        self.num_available_kv_blocks = int(available_mem//block_size_bytes)
        assert self.num_available_kv_blocks >= 1, f"Not enough memory to hold even one kv cache block on rank {self.rank}."
    
        # ===== (4) Allocate memory for kv cache. =====
        # Although `allocated_kv_cache` is a local variable, it will not be deleted out of the function,
        # because it will be referred by kv cache variables in model layers.
        allocated_kv_cache = torch.empty(2, num_layers, self.num_available_kv_blocks, self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')

        # ===== (5) Divide the giant kv cache pool into blocks and assign blocks to layers in model. =====
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
    
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare the data for prefill forward pass. """
        
        # All uncached token ids.
        # shape: (sum of number of uncached tokens,)
        input_ids = []
        positions = []
        # Where the cache of each uncached token should be written to.
        # shape: (sum of number of uncached tokens,)
        slot_mapping = []
        
        # Number of all tokens in each sequence, considering both cached and uncached tokens.
        # shape: (number of sequences,)
        seqlens_q = []
        # Cumulative sum of `seqlens_q`.
        # shape: (number of sequences + 1,)
        cu_seqlens_q = [0]
        # Number of uncached tokens in each sequence.
        # shape: (number of sequences,)
        seqlens_k = []
        # Cumulative sum of `seqlens_k`.
        # shape: (number of sequences + 1,)
        cu_seqlens_k = [0]
        
        # If there are cached token, then maps sequence index to cache block indexs.
        # shape: (number of sequences, max number of blocks per sequence)
        # It can be None if there are no cached tokens.
        block_tables = []
        
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            # Combining seperate sequences into a long sequence,
            # which enables efficient processing with variable length sequences.
            input_ids.extend(token_ids[num_cached_tokens:])
            positions.extend(list(range(num_cached_tokens, len(seq))))
            seqlens_q.append(len(token_ids)-num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1]+seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1]+seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    # Check if the block is the last block of the sequence.
                    if seq.num_cached_blocks+i != seq.num_blocks-1:
                        slot_mapping.extend(list(range(block_id*self.block_size, (block_id+1)*self.block_size)))
                    else:
                        slot_mapping.extend(list(range(block_id*self.block_size, block_id*self.block_size+seq.last_block_num_tokens)))

        # The block table will be passed to Triton kernel. And Triton kernel requires all sequences have same number of blocks.
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(block_table) for block_table in all_block_tables)
            for seq in seqs:
                aligned_block_table = seq.block_table+[-1]*(max_num_blocks-len(seq.block_table))
                block_tables.append(aligned_block_table)
        
        # Allocate input token ids to pinned memory buffer, accelerating data transfer between host and device.
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            context_lens=None,
        )
        
        return input_ids, positions
    
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare the data for decode forward pass. One token per sequence. """
        
        input_ids = []
        positions = []
        slot_mapping = []
        block_tables = []
        # Number of handled tokens in each sequence.
        # shape: (number of sequences,)
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1]*self.block_size+seq.last_block_num_tokens-1)
        
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(block_table) for block_table in all_block_tables)
        for seq in seqs:
            aligned_block_table = seq.block_table+[-1]*(max_num_blocks-len(seq.block_table))
            block_tables.append(aligned_block_table)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            context_lens=torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
        )
        
        return input_ids, positions
    
    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare sampling temperature for each sequence. """
        
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        Run model inference for a batch of sequences and return logits.
        In decode phase, replaying CUDA graph to accelerate inference.
        """
        
        # In two cases, we can not use CUDA graph:
        # (1) Prefill phase. In prefill phase, length and structure of sequence varies greatly,
        #     but CUDA graph requires fixed computation graph and memory assess mode.
        # (2) Enforce eager mode. CUDA graph does not support manual control flow.
        if is_prefill or self.enforce_eager:
            logits = self.model.compute_logits(self.model(input_ids, positions))
        # Use CUDA graph for decode phase.
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
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """ Run model inference for a batch of sequences and return output token ids. """
        
        # Prepare the data for forward pass (prefill or decode).
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            
        # Prepare sampling temperatures.
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # Execute model inference.
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # Sample tokens from logits.
        # Convert token ids to list of int, since sequence only supports int token ids.
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # Reset context for next forward pass.
        reset_context()
        
        return token_ids