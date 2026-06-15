import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from babyvllm.utils import get_context


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer maps the input indices to the corresponding embeddings.
    Parallel embedding layer splits the embeddings across multiple devices.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Args:
            num_embeddings: The number of embeddings.
            embedding_dim: The dimension of the embeddings.
        """
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        
        self.num_embeddings = num_embeddings
        # Each device should store the same number of part embeddings,
        # so pad the number of embeddings to be divisible by `tp_size`.
        self.padded_num_embeddings = (num_embeddings+self.tp_size-1)//self.tp_size*self.tp_size
        self.num_embeddings_per_partition = self.padded_num_embeddings//self.tp_size
        self.embedding_dim = embedding_dim
        
        # The start and end index of the vocabulary on the current device.
        self.vocab_start_idx = self.tp_rank*self.num_embeddings_per_partition
        self.vocab_end_idx = min(self.vocab_start_idx+self.num_embeddings_per_partition, self.num_embeddings)
        
        # weight shape: (num_embeddings_per_partition, embedding_dim)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: nn.Parameter):
        """
        Load the weight from the checkpoint.
        Args:
            param: The parameter to load the weight to.
            loaded_weights: The weight to load.
        """
        param_data = param.data
        
        shard_size = self.vocab_end_idx-self.vocab_start_idx
        slided_weights = loaded_weights.narrow(0, self.vocab_start_idx, shard_size)
        # Load the weight to the current device.
        # If needed, pad the weight with zeros.
        param_data[:shard_size] = slided_weights
        if shard_size < self.num_embeddings_per_partition:
            param_data[shard_size:].zeros_()
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: The input tensor of shape (total_tokens,).
        Returns:
            The output tensor of shape (total_tokens, embedding_dim).
        """
        if self.tp_size > 1:
            # Filter out the indices that are responsible for the current device.
            # mask shape: (total_tokens,)
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # `x-self.vocab_start_idx` convert the global token ID to the local index on the current device.
            # For example, the global token ID is 1200, the beginning index of the current device is 1000,
            # so the local index on the current device is 1200-1000=200.
            # Tokens that are not responsible for the current device will be masked to 0, which is temporary placeholder,
            # and will be filtered out in the next step.
            x = mask*(x-self.vocab_start_idx)
            
        # output shape: (total_tokens, embedding_dim)
        output = F.embedding(x, self.weight)
        
        # The output elements of "mask == False" and "x == self.vocab_start_idx" are all `self.weight[0]`,
        # so we need to distinguish them and filter the temporary placeholder generate by "mask == False".
        if self.tp_size > 1:
            output = mask.unsqueeze(-1)*output
            # Embedding is an input-side vocab-parallel layer. Each token ID is
            # owned by exactly one vocab shard, so non-owner ranks contribute
            # zeros and an all-reduce SUM restores the full hidden state on
            # every TP rank for the following transformer layers.
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        return output
        

class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Args:
            num_embeddings: The number of embeddings.
            embedding_dim: The dimension of the embeddings.
        """
        super().__init__(num_embeddings, embedding_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: The input tensor of shape (total_tokens, embedding_dim).
        Returns:
            The logits tensor of shape (batch_size, num_embeddings).
        """
        context = get_context()
        if context.is_prefill:
            # shape of input tensor is (total_tokens, embedding_dim).
            # We only need the logits of the last token in each sequence.
            last_token = context.cu_seqlens_q[1:]-1
            x = x[last_token].contiguous()
        
        # `F.linear` automatically transpose the weight matrix.
        # `self.weight` is derived from the parallel embedding layer, which achieves parameter sharing.
        # logits shape: (total_tokens, num_embeddings_per_partition)
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # LM head is an output-side vocab-parallel layer. Each rank computes
            # logits for only its vocab shard, so the full logits are formed by
            # concatenating shards, not summing them. Only rank 0 needs the full
            # vocab logits because sampling is owned by rank 0.
            all_logits = None
            if self.tp_rank == 0:
                all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)]
            dist.gather(logits, all_logits, dst=0)
            
            # Concatenate the logits from all GPUs and trim the extra embeddings.
            if self.tp_rank == 0:
                # logits shape: (total_tokens, padded_num_embeddings)
                logits = torch.cat(all_logits, dim=-1)
                # logits shape: (total_tokens, num_embeddings)
                logits = logits[:, :self.num_embeddings]
            
        # No need to broadcast the logits to all GPUs, since the main GPU will execute the sampling.
        
        return logits
    
