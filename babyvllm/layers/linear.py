import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    Base class for linear layers with tensor parallelism.
    
    Args:
        input_size: The size of the input features.
        output_size: The size of the output features.
        bias: Whether to include a bias term. Defaults to True.
        tp_dim: The dimension to parallelize. Defaults to None.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int|None = None,
    ):
        """
        Args:
            input_size: The size of the input features.
            output_size: The size of the output features.
            bias: Whether to include a bias term. Defaults to True.
            tp_dim: The dimension to parallelize. Defaults to None.
        """
        super().__init__()
        self.tp_dim = tp_dim
        # `tp_rank` is the rank of the current device
        self.tp_rank = dist.get_rank()
        # `tp_size` is the total number of devices
        self.tp_size = dist.get_world_size()
        
        # create weight and bias parameters with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        Load a saved model checkpoint.
        
        Args:
            param: The parameter to load the weight into.
            loaded_weights: The weight tensor loaded from the checkpoint.
        
        The core calling sequence of `weight_loader` is as follows:
        for name, param in model.named_parameters():
            if name in checkpoint:
                # full model parameter
                loaded_weights = checkpoint[name]
                
                # check if the parameter has a custom weight_loader
                if hasattr(param, 'weight_loader'):
                    # call custom weight_loader
                    param.weight_loader(param, loaded_weights)
                    # weight_loader will automatically:
                        # 1. extract the shard corresponding to the current GPU
                        # 2. copy it to param.data
                else:
                    # default: copy directly
                    param.data.copy_(loaded_weights)
        """
        raise NotImplementedError("weight_loader must be implemented in subclass.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward must be implemented in subclass.")
    
class ReplicatedLinear(LinearBase):
    """
    The simplest linear layer which does not add more parallelism than the original linear layer.
    Simply copies the weight from the checkpoint.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        """
        Args:
            input_size: The size of the input features.
            output_size: The size of the output features.
            bias: Whether to include a bias term. Defaults to True.
        """
        super().__init__(input_size, output_size, bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)
    
    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
    """
    ColumnParallelLinear is a linear layer where the output dimension is parallelized across multiple devices.
    Each device only gets a shard of the output features, and the input features are replicated.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        """
        Args:
            input_size: The size of the input features.
            output_size: The size of the output features.
            bias: Whether to include a bias term. Defaults to True.
        """
        tp_size = dist.get_world_size()
        # Output dimension must be divisible by the number of devices
        # to ensure each device gets equal share of the output features.
        assert output_size%tp_size == 0, f"Output size {output_size} must be divisible by tensor parallel size {tp_size}"
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        Load the weight from the checkpoint.
        Args:
            param: The parameter to load the weight to.
            loaded_weights: The weight to load.
        """
        # Calculate the shard size and check if it matches the initialized parameter data size.
        param_data = param.data
        full_data_output_size = loaded_weights.size(0)
        shard_size = full_data_output_size//self.tp_size
        assert shard_size == param_data.size(0), f"Shard size {shard_size} must be equal to parameter data size {param_data.size(0)}"
        # Copy the shard corresponding to the current GPU to the parameter data.
        start_idx = self.tp_rank*shard_size
        slided_weight = loaded_weights.narrow(0, start_idx, shard_size)
        param_data.copy_(slided_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class QKVColumnParallelLinear(ColumnParallelLinear):
    """
    QKVColumnParallelLinear is a linear layer where the output dimension is parallelized across multiple devices.
    It contains part of the q, k, and v weights in a single linear layer.
    """
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int|None = None,
        bias: bool = False,
    ):
        """
        Args:
            input_size: The size of the input features.
            head_size: The size of each head.
            num_heads: The number of heads.
            num_kv_heads: The number of kv heads. Defaults to None.
            bias: Whether to include a bias term. Defaults to False.
        """
        self.tp_size = dist.get_world_size()
        self.head_size = head_size
        self.num_heads = num_heads//self.tp_size
        self.num_kv_heads = num_kv_heads//self.tp_size if num_kv_heads is not None else self.num_heads
        total_output_size = head_size*(num_heads+2*num_kv_heads)
        super().__init__(input_size, total_output_size, bias)
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weights: torch.Tensor,
        loaded_weight_id: str,
    ):
        """
        Load the q, k, or v weight from the checkpoint.
        Args:
            param: The parameter to load the weight to.
            loaded_weights: The weight to load.
            loaded_weight_id: Whether to load the q, k, or v weight.
        """
        param_data = param.data
        assert loaded_weight_id in ['q', 'k', 'v'], f"Loaded weight id {loaded_weight_id} must be in ['q', 'k', 'v']"
        # The offset depends on the loaded weight id, since qkv are concatenated.
        if loaded_weight_id == 'q':
            offset = 0
            shard_size = self.head_size*self.num_heads
        elif loaded_weight_id == 'k':
            offset = self.head_size*self.num_heads
            shard_size = self.head_size*self.num_kv_heads
        else:
            offset = self.head_size*self.num_heads+self.head_size*self.num_heads
            shard_size = self.head_size*self.num_kv_heads
        
        # Copy the shard corresponding to the current GPU to the parameter data.
        slided_weights_start_idx = self.tp_rank*shard_size
        slided_weights = loaded_weights.narrow(0, slided_weights_start_idx, shard_size)
        param_data = param_data.narrow(0, offset, shard_size)
        param_data.copy_(slided_weights)

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
    ):
        """
        Args:
            input_size: The size of the input features.
            output_sizes: The sizes of several output features.
            bias: Whether to include a bias term. Defaults to True.
        """
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weights: torch.Tensor,
        loaded_weight_id: int,
    ):
        """
        Load the weight from the checkpoint.
        Args:
            param: The parameter to load the weight to.
            loaded_weights: The weight to load.
            loaded_weight_id: The id of the loaded weight.
            
        For example, gate projection and up projection in MLP are merged into a single linear layer.
        checkpoint = {
            'mlp.gate_proj.weight': torch.randn(intermediate_size, hidden_size),
            'mlp.up_proj.weight': torch.randn(intermediate_size, hidden_size),
            'mlp.down_proj.weight': torch.randn(hidden_size, intermediate_size),
        }
        load to
        merged_layer = Linear(
            intput_size=intermediate_size,
            output_sizes=sum([intermediate_size, intermediate_size]), # gate and up
        )
        """
        param_data = param.data
        offset = sum(self.output_sizes[:loaded_weight_id])//self.tp_size
        shard_size = self.output_sizes[loaded_weight_id]//self.tp_size
        # split along the output dimension
        param_data = param_data.narrow(0, offset, shard_size)
        loaded_weights_start_idx = self.tp_rank*shard_size
        slided_weights = loaded_weights.narrow(0, loaded_weights_start_idx, shard_size)
        param_data.copy_(slided_weights)
        

class RowParallelLinear(LinearBase):
    """
    RowParallelLinear is a linear layer where the input dimension is parallelized across multiple devices.
    Each device only gets a shard of the input features, and the output features are replicated.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        """
        Args:
            input_size: The size of the input features.
            output_size: The size of the output features.
            bias: Whether to include a bias term. Defaults to True.
        """
        tp_size = dist.get_world_size()
        # Input dimension must be divisible by the number of devices
        # to ensure each device gets equal share of the input features.
        assert input_size%tp_size == 0, f"Input size {input_size} must be divisible by tensor parallel size {tp_size}"
        super().__init__(input_size//tp_size, output_size, bias, tp_dim=1)
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        Load the weight from the checkpoint.
        Args:
            param: The parameter to load the weight to.
            loaded_weights: The weight to load.
        """
        # Calculate the shard size and check if it matches the initialized parameter data size.
        param_data = param.data
        full_data_input_size = loaded_weights.size(1)
        shard_size = full_data_input_size//self.tp_size
        assert shard_size == param_data.size(1), f"Shard size {shard_size} must be equal to parameter data size {param_data.size(1)}"
        # Copy the shard corresponding to the current GPU to the parameter data.
        start_idx = self.tp_rank*shard_size
        slided_weight = loaded_weights.narrow(1, start_idx, shard_size)
        param_data.copy_(slided_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Row parallel layer usually follows a column parallel layer.
        # `x` is the output of the column parallel layer.
        result = F.linear(x, self.weight, self.bias)
        # Reduce the output across all devices.
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result
    
