import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers.rwkv6 import LoRA as rwkvLoRA
from quantize.quantizer import UniformAffineQuantizer






class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class QuantLoRA(nn.Module):
    def __init__(
        self,
        org_module: rwkvLoRA,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        
        # Initialize from existing rwkvLoRA module if provided
        self.input_dim = org_module.input_dim
        self.output_dim = org_module.output_dim
        self.low_rank_dim = org_module.low_rank_dim
        self.bias = org_module.bias
        self.activation = org_module.activation

        self.lora = nn.Sequential(
            QuantLinear(org_module.lora[0], weight_quant_params, act_quant_params, disable_input_quant),
            self.activation,
            QuantLinear(org_module.lora[2], weight_quant_params, act_quant_params, disable_input_quant),
        )

        # Quantization state flags
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_input_quant = disable_input_quant
            
    def forward(self, x: torch.Tensor):
        out = self.lora(x)
        return out

class SmoothLinear(nn.Module):
    """
    No quantization, only smooth
    """
    def __init__(
        self,
        org_module: nn.Linear,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.use_temporary_parameter = False
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return out
