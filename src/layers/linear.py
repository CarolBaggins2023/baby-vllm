import torch
import torch.nn as nn
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
        super().__init__()
        # `tp_dim` is the dimension to parallelize
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
            register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: nn.Parameter):
        """
        Load a saved model checkpoint.
        
        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor loaded from the checkpoint.
        
        The core calling sequence of `weight_loader` is as follows:
        for name, param in model.named_parameters():
            if name in checkpoint:
                # full model parameter
                loaded_weight = checkpoint[name]
                
                # check if the parameter has a custom weight_loader
                if hasattr(param, 'weight_loader'):
                    # call custom weight_loader
                    param.weight_loader(param, loaded_weight)
                    # weight_loader will automatically:
                        # 1. extract the shard corresponding to the current GPU
                        # 2. copy it to param.data
                else:
                    # default: copy directly
                    param.data.copy_(loaded_weight)
        """
        raise NotImplementedError("weight_loader must be implemented in subclass.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward must be implemented in subclass.")