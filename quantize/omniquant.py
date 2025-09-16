import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.omni_norm import OmniLayerNorm
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    is_rwkv7 = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    elif 'rwkv' in args.net.lower():
        is_rwkv7 = True
        layers = model.model.layers
        model.model.embeddings = model.model.embeddings.to(dev)
        model.lm_head = model.lm_head.to(dev)
        layer_name_prefix = "model.layers"
        pairs = {
            # "r_proj":"qkv",
            "key":"fc1"
        }
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    inps_vfirst = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_rwkv7 = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            if self.is_rwkv7:
                outputs = self.module(inp, **kwargs)
                inps_vfirst[cache["i"]] = outputs[-1].cpu()
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_rwkv7 = is_rwkv7

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    elif 'rwkv' in args.net.lower():
        model.model.embeddings = model.model.embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    print(f"layers: {layers[0]}")
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        elif 'rwkv' in args.net.lower():
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "lora" in name:
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)
                # replace LayerNorm with OmniLayerNorm
                if isinstance(module, torch.nn.LayerNorm):
                    omnilayernorm = OmniLayerNorm(module)
                    add_new_module(name, qlayer, omnilayernorm)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(0, args.nsamples, args.batch_size):
                        index = j
                        end_index = min(index + args.batch_size, args.nsamples)
                        current_batch_size = end_index - index
                        
                        current_attention_mask = attention_mask_batch[:current_batch_size] if attention_mask_batch is not None else None

                        if is_rwkv7:
                            fp_inps[index:end_index] = qlayer(fp_inps[index:end_index].to(dev), attention_mask=current_attention_mask,position_ids=position_ids, v_first=inps_vfirst[index:end_index].to(dev))[0].cpu()
                            if args.aug_loss:
                                fp_inps_2[index:end_index] = qlayer(quant_inps[index:end_index].to(dev), attention_mask=current_attention_mask,position_ids=position_ids, v_first=inps_vfirst[index:end_index].to(dev))[0].cpu()
                        else:
                            fp_inps[index:end_index] = qlayer(fp_inps[index:end_index].to(dev), attention_mask=current_attention_mask,position_ids=position_ids)[0].cpu()
                            if args.aug_loss:
                                fp_inps_2[index:end_index] = qlayer(quant_inps[index:end_index].to(dev), attention_mask=current_attention_mask,position_ids=position_ids)[0].cpu()
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or is_rwkv7 or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            if is_rwkv7:
                # qlayer.register_parameter("qkv_smooth_scale",torch.nn.Parameter(torch.ones(layer.attn.r_proj.in_features,device=dev, dtype=dtype)))
                # qlayer.register_parameter("qkv_smooth_shift",torch.nn.Parameter(torch.zeros_like(qlayer.qkv_smooth_scale)))
                qlayer.register_parameter("rkv_smooth_scale",torch.nn.Parameter(torch.ones(layer.attn.r_proj.in_features,device=dev, dtype=dtype)))
                qlayer.register_parameter("rkv_smooth_shift",torch.nn.Parameter(torch.zeros_like(qlayer.rkv_smooth_scale)))
                qlayer.register_parameter("fc1_smooth_scale",torch.nn.Parameter(torch.ones(layer.ffn.key.in_features,device=dev, dtype=dtype)))
                qlayer.register_parameter("fc1_smooth_shift",torch.nn.Parameter(torch.zeros_like(qlayer.fc1_smooth_scale)))
            else:
            # if not is_rwkv7:
                qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
                for name,module in qlayer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in pairs.keys():
                            if key in name:
                                act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                                weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                                scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                                print(name, scale)
                                if use_shift and not is_llama and not is_rwkv7:
                                    shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                                else:
                                    shift = torch.zeros_like(scale)
                                qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                                qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama, is_rwkv7)
                        if is_rwkv7:
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_ids=position_ids, v_first=inps_vfirst[index:index+args.batch_size].to(dev))[0]
                        else:
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,].to(dev), quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,].to(dev), quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama, is_rwkv7)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(0, args.nsamples, args.batch_size):
                        index = j
                        end_index = min(index + args.batch_size, args.nsamples)
                        current_batch_size = end_index - index
                        
                        current_attention_mask = attention_mask_batch[:current_batch_size] if attention_mask_batch is not None else None

                        if is_rwkv7:
                            quant_inps[index:end_index] = qlayer(quant_inps[index:end_index].to(dev), attention_mask=current_attention_mask,position_ids=position_ids, v_first=inps_vfirst[index:end_index].to(dev))[0].cpu()
                        else:
                            quant_inps[index:end_index] = qlayer(quant_inps[index:end_index].to(dev), attention_mask=current_attention_mask,position_ids=position_ids)[0].cpu()
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    if is_rwkv7 and getattr(args, 'quant_lm_head', False):
        logger.info(f"=== Start quantize lm_head ===")
        lm_head = model.lm_head.to(dev)
        q_lm_head = QuantLinear(lm_head, args.weight_quant_params, args.act_quant_params).to(dev)
        lm_norm = OmniLayerNorm(model.model.norm).to(dev)
        q_lm_head.let = args.let
        if args.let:
            q_lm_head.register_parameter("fc1_smooth_scale",torch.nn.Parameter(torch.ones(lm_head.in_features,device=dev, dtype=dtype)))
            q_lm_head.register_parameter("fc1_smooth_shift",torch.nn.Parameter(torch.zeros_like(q_lm_head.fc1_smooth_scale)))

        if args.epochs > 0:
            fp_lm_head_out = torch.zeros(
                (args.nsamples, lm.seqlen, model.config.vocab_size), dtype=dtype, device='cpu'
            )
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(0, args.nsamples, args.batch_size):
                        index = j
                        end_index = min(index + args.batch_size, args.nsamples)
                        fp_lm_head_out[index:end_index] = lm_head(quant_inps[index:end_index].to(dev)).cpu()

        set_quant_state(q_lm_head, weight_quant=False, act_quant=True)

        if args.epochs > 0:
            with torch.no_grad():
                q_lm_head.float()
            use_shift = False
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(q_lm_head, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    with traincast():
                        smooth_and_quant_temporary(q_lm_head, args, False, True, True, lm_norm)
                        quant_out = q_lm_head(quant_inps[index:index+args.batch_size,].to(dev))
                        loss = loss_func(fp_lm_head_out[index:index+args.batch_size,].to(dev), quant_out)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=lwc_parameters(q_lm_head)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"lm_head iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(q_lm_head)
            del optimizer
        
        q_lm_head.half()
        lm_norm.half()
        smooth_and_quant_inplace(q_lm_head, args, False, True, True, lm_norm)
        
        if args.epochs > 0:
            register_scales_and_zeros(q_lm_head)
            model.lm_head = q_lm_head.to("cpu")
            omni_parameters['lm_head'] = omni_state_dict(q_lm_head)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(q_lm_head)
            model.lm_head = q_lm_head.to("cpu")

        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16
            named_linears = get_named_linears(q_lm_head)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                
                if name == '':
                    q_lm_head = q_linear
                else:
                    add_new_module(name, q_lm_head, q_linear)
                print(f"pack quantized lm_head finished")
                del module
        model.lm_head = q_lm_head.to("cpu")
        model.model.norm = lm_norm.to("cpu")
        del lm_head
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

