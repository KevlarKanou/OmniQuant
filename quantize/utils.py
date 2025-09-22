from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama, isrwkv7, is_lm_head=False, lm_norm=None):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        elif isrwkv7:
            if is_lm_head and lm_norm is not None:
                smooth_ln_fcs_temporary(lm_norm, [model],
                                        model.fc1_smooth_scale,model.fc1_smooth_shift)
            else:
                rkv = [model.attn.r_proj, model.attn.k_proj, model.attn.v_proj]
                
                # if args.quant_lora:
                    # Numerical instability...
                    # rkv += [model.attn.w_lora.lora[0], model.attn.a_lora.lora[0], model.attn.g_lora.lora[0]]
                    # if hasattr(model.attn, "v_lora"):
                    #     rkv += [model.attn.v_lora.lora[0]]
                if args.quant_lora:
                    model.attn.w_lora.lora[0].temp_weight = model.attn.w_lora.lora[0].weight
                    model.attn.w_lora.lora[2].temp_weight = model.attn.w_lora.lora[2].weight
                    model.attn.g_lora.lora[0].temp_weight = model.attn.g_lora.lora[0].weight
                    if args.lora_smooth:
                        smooth_fc_fc_temporary(model.attn.a_lora.lora[0],model.attn.a_lora.lora[2],
                                model.lora_a_smooth_scale, model.lora_a_smooth_shift)
                        if hasattr(model.attn, "v_lora"):
                            smooth_fc_fc_temporary(model.attn.v_lora.lora[0],model.attn.v_lora.lora[2],
                                model.lora_v_smooth_scale, model.lora_v_smooth_shift)
                    else:
                        model.attn.a_lora.lora[0].temp_weight = model.attn.a_lora.lora[0].weight
                        model.attn.a_lora.lora[2].temp_weight = model.attn.a_lora.lora[2].weight
                        if hasattr(model.attn, "v_lora"):
                            model.attn.v_lora.lora[0].temp_weight = model.attn.v_lora.lora[0].weight
                            model.attn.v_lora.lora[2].temp_weight = model.attn.v_lora.lora[2].weight
                    if args.o_proj_smooth:
                        smooth_fc_fc_temporary(model.attn.g_lora.lora[2],model.attn.o_proj,
                                model.out_smooth_scale)
                    else:
                        model.attn.g_lora.lora[2].temp_weight = model.attn.g_lora.lora[2].weight
                        model.attn.o_proj.temp_weight = model.attn.o_proj.weight
                else:
                    model.attn.o_proj.temp_weight = model.attn.o_proj.weight

                smooth_ln_fcs_temporary(model.attn_norm, rkv,
                                        model.rkv_smooth_scale,model.rkv_smooth_shift)
                smooth_ln_fcs_temporary(model.ffn_norm, [model.ffn.key],
                                        model.fc1_smooth_scale,model.fc1_smooth_shift)
                if args.cmix_kv_smooth:
                    smooth_cmix_k_v_temporary(model.ffn.key, model.ffn.value, model.fc2_smooth_scale)
                else:
                    model.ffn.value.temp_weight = model.ffn.value.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight

    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama, isrwkv7, is_lm_head=False, lm_norm=None):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        elif isrwkv7:
            if is_lm_head and lm_norm is not None:
                smooth_ln_fcs_inplace(lm_norm, [model],
                                        model.fc1_smooth_scale,model.fc1_smooth_shift)
            else:
                rkv = [model.attn.r_proj, model.attn.k_proj, model.attn.v_proj]
                # Numerical instability...
                # if args.quant_lora:
                #     rkv += [model.attn.w_lora.lora[0], model.attn.a_lora.lora[0], model.attn.g_lora.lora[0]]
                #     if hasattr(model.attn, "v_lora"):
                #         rkv += [model.attn.v_lora.lora[0]]
                if args.quant_lora and args.lora_smooth:
                    smooth_fc_fc_inplace(model.attn.a_lora.lora[0],model.attn.a_lora.lora[2],
                            model.lora_a_smooth_scale, model.lora_a_smooth_shift)
                    if hasattr(model.attn, "v_lora"):
                        smooth_fc_fc_inplace(model.attn.v_lora.lora[0],model.attn.v_lora.lora[2],
                            model.lora_v_smooth_scale, model.lora_v_smooth_shift)
                
                if args.quant_lora and args.o_proj_smooth:
                    smooth_fc_fc_inplace(model.attn.g_lora.lora[2],model.attn.o_proj,
                        model.out_smooth_scale)

                smooth_ln_fcs_inplace(model.attn_norm, rkv,
                                        model.rkv_smooth_scale,model.rkv_smooth_shift)
                smooth_ln_fcs_inplace(model.ffn_norm, [model.ffn.key],
                                        model.fc1_smooth_scale,model.fc1_smooth_shift)
                if args.cmix_kv_smooth:
                    smooth_cmix_k_v_inplace(model.ffn.key, model.ffn.value, model.fc2_smooth_scale)

        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        if not isrwkv7:
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
