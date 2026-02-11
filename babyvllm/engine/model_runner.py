import torch

from babyvllm.models.qwen3 import Qwen3ForCausalLM
from babyvllm.layers.sampler import Sampler
from babyvllm.engine.sequence import Sequence
from babyvllm.utils.context import set_context

class ModelRunner:
    """
    ModelRunner acts as a bridge between sequences and model execution.
    It is responsible for:
    (1) Data preparation: `prepare_prefill`, `prepare_decode`, `prepare_sample`.
    (2) Memory management: `warmup_model`, `allocate_kv_cache`.
    (3) Model execution: `run_model`, `run`.
    (4) Shared memory communication: `read_shm`, `write_shm`.
    (5) CUDA graph optimization: `capture_cudagraph`.
    """
    
    def __init__(self, config: dict, rank: int):
        self.config = config
        
        # Set distributed config.
        self.world_size = config['world_size']
        self.block_size = config['block_size']
        
        self.rank = rank
        
        # model creation
        self.model = Qwen3ForCausalLM(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_heads=config['num_heads'],
            head_dim=config['head_dim'],
            scale=config['scale'],
            num_kv_heads=config['num_kv_heads'],
            rms_norm_epsilon=config['rms_norm_epsilon'],
            qkv_bias=config['qkv_bias'],
            base=config['base'],
            max_position=config['max_position'],
            intermediate_size=config['intermediate_size'],
            ffn_bias=config['ffn_bias'],
            num_layers=config['num_layers'],
            tie_word_embeddings=config['tie_word_embeddings'],
        ).cuda(rank)
        self.sampler = Sampler()
        
        # Get peak memory usage, which is helpful for kv cache allocation.
        self.warmup_model()
        
        # Allocate kv cache.
        self.default_dtype = torch.get_default_dtype()
        self.allocate_kv_cache()
        
    def warmup_model(self):
        """ Warmup the model and record the peak memory usage. """
        
        # Cleanup memory pool of PyTorch. It will free unused memory.
        torch.cuda.empty_cache()
        # Reset peak memory usage stats.
        torch.cuda.reset_peak_memory_stats()
        
        # Create a batch of fake sequences and run the model.
        max_tokens = self.config['max_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length) for _ in range(batch_size)]
        self.run(seqs=seqs, is_prefill=True)
        torch.cuda.empty_cache()
    
    def allocate_kv_cache(self):
        # ===== (1) Find available memory while reserving room for model execution. =====
        # Get free memory and total memory of the current device.
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem*self.config['gpu_memory_utilization']
        # Reserve room for model execution.
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        available_mem = total_free_mem-(peak_mem_usage-current_mem_usage)
        
        # ===== (2) Find parameters to compute kv cache size. =====
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads']
        head_dim = self.config['head_dim']
        
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
    
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare the data for prefill forward pass. """
        
        # All uncached token ids.
        # shape: (sum of number of uncached tokens,)
        input_ids = []
        # Positions of each uncached token.
        # shape: (sum of number of uncached tokens,)
        positions = []
        # Where the cache of each uncached token should be written to.
        # shape: (sum of number of uncached tokens,)
        slot_mappings = []
        
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
            input_ids.extend([token_ids[num_cached_tokens:]])
            positions.extend(list(range(num_cached_tokens, len(seq))))
            seqlens_q.append(len(token_ids)-num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1]+seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1]+seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    # Check if the block is the last block of the sequence.
                    if seq.num_cached_blocks+i != seq.num_blocks-1:
                        slot_mappings.extend(list(range(block_id*self.block_size, (block_id+1)*self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id*self.block_size, seq.last_block_num_tokens)))

        # The block table will be passed to Triton kernel. And Triton kernel requires all sequences have same number of blocks.
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(block_table) for block_table in all_block_tables)
            for seq in seqs:
                aligned_block_table = seq.block_table+[-1]*(max_num_blocks-len(seq.block_table))
                block_tables.append(aligned_block_table)
        
        # Allocate input token ids to pinned memory buffer, accelerating data transfer between host and device.
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mappings=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            context_lens=None,
        )
        
        return input_ids, positions
    
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare the data for decode forward pass. One token per sequence. """
        
        input_ids = []
        positions = []
        slot_mappings = []
        block_tables = []
        # Number of handled tokens in each sequence.
        # shape: (number of sequences,)
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1]*self.block_size+seq.last_block_num_tokens-1)
        
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(block_table) for block_table in all_block_tables)
        for seq in seqs:
            aligned_block_table = seq.block_table+[-1]*(max_num_blocks-len(seq.block_table))
            block_tables.append(aligned_block_table)

        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mappings=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
        )
        
        return input_ids, positions
    
    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        """ Prepare sampling temperature for each sequence. """
        
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        pass
    