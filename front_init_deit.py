import os
import torch
import torch.nn as nn
from torch.fft import fft, ifft
import argparse
from pathlib import Path
import des_models

# ==========================================
# 1. Core Utility Functions and Classes (DCT, IDCT, ToDCT, etc.)
# ==========================================
def dct(x):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * torch.pi / (2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if len(x_shape) == 1:
        V = V[0]
    else:
        V = V.view(*x_shape)
    return V

def idct(X):
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, N)
    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * torch.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=N, dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    if len(x_shape) == 1:
        x = x[0]
    else:
        x = x.view(*x_shape)
    return x

class ToDCT(nn.Module):
    # Added mlp_dct_dim parameter to control 1D/2D processing
    def __init__(self, keep_ratio=0.4, model_name='cnn3', mlp_dct_dim='2d'):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.layer_blocks_norm = {}
        self.layer_blocks_freq = {}
        self.layer_presevered_freq = {}
        self.model_name = model_name
        self.mlp_dct_dim = mlp_dct_dim

    def normalize_weight(self, weight_matrix):
        mean = weight_matrix.mean()
        std = weight_matrix.std()
        normalized_weight = (weight_matrix - mean) / (std + 1e-8)
        return normalized_weight, mean.item(), std.item()

    # Pass layer_name to identify if it is fc1/fc2
    def _process_single_layer(self, block, layer_name=""):
        normalized_block, mean, std = self.normalize_weight(block)
        stats = {'mean': mean, 'std': std}

        if normalized_block.dim() >= 2:
            if normalized_block.shape[0] == 1 or normalized_block.shape[1] == 1: 
                normalized_block = normalized_block.squeeze()

        if normalized_block.dim() == 1:
            freq_coeffs = dct(normalized_block)
            length = int(len(freq_coeffs) * self.keep_ratio)
            freq_coeffs_compressed = freq_coeffs[:length]
            freq_loss = torch.sum(freq_coeffs[length:] ** 2)
            presevered_freq = torch.sum(freq_coeffs[:length] ** 2)
        else:
            is_fc1 = 'mlp.fc1' in layer_name
            is_fc2 = 'mlp.fc2' in layer_name

            # If 1D mode is enabled and it hits an MLP layer
            if self.mlp_dct_dim == '1d' and (is_fc1 or is_fc2):
                if is_fc1:
                    # FC1 shape: [hidden_dim, embed_dim], sort along dim=0
                    # Perform 1D DCT only along dim=0 (row direction)
                    freq_coeffs = dct(normalized_block.T).T
                    h_keep = int(freq_coeffs.shape[0] * self.keep_ratio)
                    freq_coeffs_compressed = freq_coeffs[:h_keep, :]
                    freq_loss = torch.sum(freq_coeffs[h_keep:, :] ** 2)
                    presevered_freq = torch.sum(freq_coeffs[:h_keep, :] ** 2)
                else:
                    # FC2 shape: [embed_dim, hidden_dim], sort along dim=1
                    # Perform 1D DCT only along dim=1 (column direction)
                    freq_coeffs = dct(normalized_block)
                    w_keep = int(freq_coeffs.shape[1] * self.keep_ratio)
                    freq_coeffs_compressed = freq_coeffs[:, :w_keep]
                    freq_loss = torch.sum(freq_coeffs[:, w_keep:] ** 2)
                    presevered_freq = torch.sum(freq_coeffs[:, :w_keep] ** 2)
            else:
                # Default original 2D-DCT processing
                freq_coeffs = dct(dct(normalized_block).T).T
                h, w = freq_coeffs.shape
                h_keep = int(h * self.keep_ratio)
                w_keep = int(w * self.keep_ratio)
                freq_coeffs_compressed = freq_coeffs[:h_keep, :w_keep]
                freq_loss = torch.sum(freq_coeffs[h_keep:, :] ** 2) + torch.sum(freq_coeffs[:, w_keep:] ** 2) - torch.sum(freq_coeffs[h_keep:, w_keep:] ** 2)
                presevered_freq = torch.sum(freq_coeffs[:h_keep, :w_keep] ** 2)
                
        return freq_loss, presevered_freq, stats, freq_coeffs_compressed

    def frequency_regularization(self, weight_matrix, layer_name):
        block_stats = []
        block_freqs = []
        freq_loss = 0
        pre_freq_total = 0
        
        if weight_matrix.dim() == 4:
            OC, IC, kH, kW = weight_matrix.shape
            matrix_2d = weight_matrix.view(OC * IC, kH * kW)
        elif weight_matrix.dim() == 3:
            OC, IC, kW = weight_matrix.shape
            matrix_2d = weight_matrix.view(OC * IC, kW)
        else:
            matrix_2d = weight_matrix

        if matrix_2d.dim() == 1:
            loss, pre_freq, stats, freq_coeffs = self._process_single_layer(matrix_2d, layer_name)
            block_stats.append(stats)
            block_freqs.append({'indices': (0, len(matrix_2d)), 'stats': stats, 'freq_coeffs': freq_coeffs})
            freq_loss += loss
            pre_freq_total += pre_freq
            self.layer_blocks_norm[layer_name] = block_stats
            self.layer_blocks_freq[layer_name] = block_freqs
            self.layer_presevered_freq[layer_name] = pre_freq_total
            return freq_loss, pre_freq_total
        elif matrix_2d.dim() == 2:
            h, w = matrix_2d.shape
            loss, pre_freq, stats, freq_coeffs = self._process_single_layer(matrix_2d, layer_name)
            block_stats.append(stats)
            block_freqs.append({'indices': (0, h, 0, w), 'stats': stats, 'freq_coeffs': freq_coeffs})
            freq_loss += loss
            pre_freq_total += pre_freq
            self.layer_blocks_norm[layer_name] = block_stats
            self.layer_blocks_freq[layer_name] = block_freqs
            self.layer_presevered_freq[layer_name] = pre_freq_total
            return freq_loss, pre_freq_total
        return 0, 0

    def forward(self, model):
        freq_loss = 0
        preveserved_freq_total = 0
        self.layer_blocks_norm.clear()  
        self.layer_blocks_freq.clear()  
        self.layer_presevered_freq.clear()

        for name, param in model.named_parameters():
            if param.requires_grad:
                f_l, p_f = self.frequency_regularization(param, name)
                freq_loss += f_l
                preveserved_freq_total += p_f

        return self.layer_blocks_freq, self.layer_blocks_norm, self.layer_presevered_freq

def get_module_names(model):
    module_names = {}
    modules={}
    for name, module in model.named_parameters():
        if module.requires_grad:
            module_name_parts = name.split('.')  
            if 'patch_embed' in name:
                module_name = name
            elif len(module_name_parts) > 2:
                module_name = '.'.join(module_name_parts[2:])  
            else:
                module_name = name
            if module_name not in module_names:
                module_names[module_name] = []
            module_names[module_name].append(name)
            modules[name]=module
    return module_names,modules

def get_layerblocks_name(model):
    module_names = {}
    modules={}
    for name in model:
        module_name_parts = name.split('.') 
        if 'patch_embed' in name:
            module_name = name
        elif len(module_name_parts) > 2:
            module_name = '.'.join(module_name_parts[2:])  
        else:
            module_name = name
        if module_name not in module_names:
            module_names[module_name] = []
        module_names[module_name].append(name)
        modules[name]=model[name]
    return module_names,modules

def get_top_n_modules(module_names, layer_names, layer_freq, layer_module,):
    layer_energy = {}
    for name in module_names:
        if name in layer_names:
            if len(module_names[name]) <= len(layer_names[name]):
                n = len(module_names[name])
                freq_dict = {layer: layer_freq[layer] for layer in layer_names[name]}
                sorted_layers = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                top_n_layers = sorted_layers[:n]

                original_order = []
                for layer in layer_names[name]:
                    if layer in [x[0] for x in top_n_layers]:
                        original_order.append(layer)
                    if len(original_order) == n:
                        break
                for i, layer_name in enumerate(original_order):
                    model_name = module_names[name][i]
                    layer_energy[model_name] = layer_module[layer_name]
            else:
                for i, layer_name in enumerate(layer_names[name]):
                    if i < len(module_names[name]): 
                        model_name = module_names[name][i]
                        layer_energy[model_name] = layer_module[layer_name]
                    else:
                        break 
    return layer_energy

def apply_dct(x, dim=0):
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(perm)
    dct_result = dct(x_perm)
    return dct_result.permute(perm)

def apply_idct(x, dim=0):
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(perm)
    dct_result = idct(x_perm)
    return dct_result.permute(perm)

def get_top_n_modules_dim_dct(module_names, layer_names, layer_module, layer_blocks_norm, select_n=None, mode='zero',freq_mode='first'):
    layer_energy = {}
    layer_norms={}
    for name in module_names:
        if name in layer_names:
            if len(module_names[name])>1:
                old_layer_params = []
                for layer_name in layer_names[name]:
                    old_layer_params.append(layer_module[layer_name][0]['freq_coeffs'])

                stacked_params = torch.stack(old_layer_params, dim=0)
                stacked_params_dct = apply_dct(stacked_params, dim=0)

                if select_n is None:
                    select_n=min(len(module_names[name]),len(layer_names[name]))
                else:
                    select_n=min(len(module_names[name]),min(select_n,len(layer_names[name])))
                
                if freq_mode=='first':
                    stacked_params_dct = stacked_params_dct[:select_n]
                elif freq_mode=='last':
                    stacked_params_dct = stacked_params_dct[-select_n:]
                elif freq_mode=='middle':
                    mid=len(layer_names[name])//2-1
                    stacked_params_dct = stacked_params_dct[mid:mid+select_n]
                
                if stacked_params_dct.dim()==3:
                    l,m,n=stacked_params_dct.shape
                    if mode=='zero':
                        selected_params_dct = torch.zeros(len(module_names[name]),m,n)
                        selected_params_dct[:select_n] = stacked_params_dct[:select_n]
                    elif mode=='random':
                        mean=torch.mean(stacked_params_dct[:select_n])
                        std=torch.std(stacked_params_dct[:select_n])
                        selected_params_dct = torch.normal(mean.item(),std.item(),(len(module_names[name]),m,n))
                        selected_params_dct[:select_n] = stacked_params_dct[:select_n]
                    elif mode=='consistence':
                        selected_params_dct = torch.zeros(len(module_names[name]), m, n)
                        L = len(module_names[name])
                        base = L // select_n
                        remainder = L % select_n
                        indices = []
                        for i in range(select_n):
                            repeat = base + 1 if i < remainder else base
                            indices.extend([i] * repeat)
                        indices_tensor = torch.tensor(indices, dtype=torch.long)
                        selected_params = stacked_params_dct[:select_n]  
                        selected_params_dct.copy_(selected_params[indices_tensor])
                    elif mode=='repeat':
                        selected_params_dct = torch.zeros(len(layer_names[name]),m,n)
                        selected_params_dct[:select_n] = stacked_params_dct[:select_n]

                elif stacked_params_dct.dim()==2:
                    l,m=stacked_params_dct.shape
                    if mode=='zero':
                        selected_params_dct = torch.zeros(len(module_names[name]),m)
                        selected_params_dct[:select_n] = stacked_params_dct[:select_n]
                    elif mode=='random':
                        mean=torch.mean(stacked_params_dct[:select_n])
                        std=torch.std(stacked_params_dct[:select_n])
                        selected_params_dct = torch.normal(mean.item(),std.item(),(len(module_names[name]),m))
                        selected_params_dct[:select_n] = stacked_params_dct[:select_n]
                    elif mode=='consistence':
                        selected_params_dct = torch.zeros(len(module_names[name]), m)
                        L = len(module_names[name])
                        base = L // select_n
                        remainder = L % select_n
                        indices = []
                        for i in range(select_n):
                            repeat = base + 1 if i < remainder else base
                            indices.extend([i] * repeat)
                        indices_tensor = torch.tensor(indices, dtype=torch.long)
                        selected_params = stacked_params_dct[:select_n]  
                        selected_params_dct.copy_(selected_params[indices_tensor])
                    elif mode=='repeat':
                        selected_params_dct = torch.zeros(len(layer_names[name]),m)
                        selected_params_dct[:select_n] = stacked_params_dct[:select_n]

                selected_params_idct = apply_idct(selected_params_dct, dim=0)

                for i in range(len(module_names[name])):  
                    model_name = module_names[name][i]
                    if i < len(layer_names[name]):
                        layer_energy[model_name] = layer_module[model_name]
                        layer_norms[model_name] = layer_blocks_norm[model_name]
                        layer_last_name=model_name
                    else:
                        layer_energy[model_name]=layer_module[layer_last_name]
                        layer_norms[model_name] = layer_blocks_norm[layer_last_name]

                    layer_energy[model_name][0]['freq_coeffs'] = selected_params_idct[i]
            else:
                layer_energy[name] = layer_module[name]
                layer_norms[name] = layer_blocks_norm[name]
    return layer_energy,layer_norms

def reconstruct_weights(model, layer_blocks_freq, layer_blocks_norm, layer_freq, use_dct=True, use_ratio=False, select_n=None, mode='zero', freq_mode='first', skip_list=['none'], mlp_dct_dim='2d'):
    reconstructed_params = {}
    module_names,model_module = get_module_names(model)
    layer_names,layer_module = get_layerblocks_name(layer_blocks_freq)
    
    if use_dct:
        selected_modules , selected_norms = get_top_n_modules_dim_dct(module_names, layer_names, layer_module, layer_blocks_norm, select_n=select_n, mode=mode, freq_mode=freq_mode)
    else:
        selected_modules  = get_top_n_modules(module_names, layer_names, layer_freq, layer_module)
        
    layer_blocks_freq = selected_modules
    layer_blocks_norm = selected_norms

    for name, param in model.named_parameters():
        if name in layer_blocks_freq:
            blocks_info = layer_blocks_freq[name]
            block_stats = layer_blocks_norm[name]

            if param.dim() == 4:  
                # (Logic for Conv remains unchanged, original truncation preservation logic)
                OC, IC, kH, kW = param.shape
                reconstructed_weight = torch.zeros_like(param)
                matrix_2d = param.view(OC * IC, kH * kW)
                freq_coeffs = blocks_info[0]['freq_coeffs']
                stats = block_stats[0]
                full_size = (matrix_2d.shape[0], matrix_2d.shape[1])
                padded_freq = torch.zeros(full_size)
                N,M=freq_coeffs.shape
                ratio_n=OC*IC/N
                ratio_m=kH*kW/M
                min_fre_h=min(freq_coeffs.shape[0],full_size[0])
                min_fre_w=min(freq_coeffs.shape[1],full_size[1])
                
                # Fix math bug: frequency domain truncation should always take the first 0:N, never slice in the middle
                padded_freq[:min_fre_h, :min_fre_w] = freq_coeffs[:min_fre_h, :min_fre_w]

                if use_ratio:
                    reconstructed_block = idct(idct(padded_freq).T).T*ratio_n*ratio_m
                else:
                    reconstructed_block = idct(idct(padded_freq).T).T
                    
                reconstructed_block = reconstructed_block * stats['std'] + stats['mean']
                reconstructed_weight.copy_(reconstructed_block.view(OC, IC, kH, kW))
                
            elif param.dim() == 3:  
                OC, IC, kW = param.shape
                reconstructed_weight = torch.zeros_like(param)
                matrix_2d = param.view(OC * IC, kW)
                freq_coeffs = blocks_info[0]['freq_coeffs']
                stats = block_stats[0]
                full_size = (matrix_2d.shape[0], matrix_2d.shape[1])
                padded_freq = torch.zeros(full_size)
                if padded_freq.dim() == 2 and freq_coeffs.dim() == 1:
                    freq_coeffs = freq_coeffs.unsqueeze(0)
                min_fre_h=min(freq_coeffs.shape[0],full_size[0])
                min_fre_w=min(freq_coeffs.shape[1],full_size[1])
                N,M=freq_coeffs.shape
                ratio_n=OC*IC/N
                ratio_m=kW/M
               
                # Fix math bug
                padded_freq[:min_fre_h, :min_fre_w] = freq_coeffs[:min_fre_h, :min_fre_w]

                if use_ratio:
                    reconstructed_block = idct(idct(padded_freq).T).T*ratio_n*ratio_m
                else:
                    reconstructed_block = idct(idct(padded_freq).T).T
                reconstructed_block = reconstructed_block * stats['std'] + stats['mean']
                reconstructed_weight.copy_(reconstructed_block.view(OC, IC, kW))
                
            else:
                if param.dim() == 1:
                    reconstructed_weight = torch.zeros_like(param)
                    freq_coeffs = blocks_info[0]['freq_coeffs']
                    stats = block_stats[0]
                    length = param.shape[0]
                    padded_freq = torch.zeros(length)
                    ratio_m=param.shape[0]/freq_coeffs.shape[0]
                    min_feq=min(freq_coeffs.shape[0],length)
                  
                    # Fix math bug
                    padded_freq[:min_feq] = freq_coeffs[:min_feq]

                    if use_ratio:
                        reconstructed_block = idct(padded_freq)*ratio_m
                    else:
                        reconstructed_block = idct(padded_freq)
                    reconstructed_block = reconstructed_block * stats['std'] + stats['mean']
                    reconstructed_weight.copy_(reconstructed_block)
                else:
                    h, w = param.shape
                    reconstructed_weight = torch.zeros((h, w))
                    freq_coeffs = blocks_info[0]['freq_coeffs']
                    if freq_coeffs.dim() == 1:
                        if h==1:
                            freq_coeffs = freq_coeffs.unsqueeze(0)
                        else:
                            freq_coeffs = freq_coeffs.unsqueeze(1)
                    stats = block_stats[0]
                    full_size = (h,w)
                    padded_freq = torch.zeros(full_size)
                    min_fre_h=min(freq_coeffs.shape[0],full_size[0])
                    min_fre_w=min(freq_coeffs.shape[1],full_size[1])
                    N,M=freq_coeffs.shape
                    ratio_n=h/N
                    ratio_m=w/M

                    # Unified fix for centered truncation bug across all 2D sizes -> always take the top-left corner (0:N)
                    padded_freq[:min_fre_h, :min_fre_w] = freq_coeffs[:min_fre_h, :min_fre_w]

                    is_fc1 = 'mlp.fc1' in name
                    is_fc2 = 'mlp.fc2' in name

                    # If 1D processing is enabled and it's an MLP layer
                    if mlp_dct_dim == '1d' and (is_fc1 or is_fc2):
                        if is_fc1:
                            # fc1 only performed DCT on dim=0 (rows)
                            if use_ratio:
                                reconstructed_block = idct(padded_freq.T).T * ratio_n
                            else:
                                reconstructed_block = idct(padded_freq.T).T
                        else:
                            # fc2 only performed DCT on dim=1 (columns)
                            if use_ratio:
                                reconstructed_block = idct(padded_freq) * ratio_m
                            else:
                                reconstructed_block = idct(padded_freq)
                    else:
                        # Original 2D-DCT
                        if use_ratio:
                            reconstructed_block = idct(idct(padded_freq).T).T*ratio_n*ratio_m
                        else:
                            reconstructed_block = idct(idct(padded_freq).T).T
                            
                    reconstructed_block = reconstructed_block * stats['std'] + stats['mean']
                    reconstructed_weight.copy_(reconstructed_block[:h, :w])
                    
            reconstructed_params[name] = reconstructed_weight
  
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in reconstructed_params:
                if name == 'head.weight' or name == 'head.bias':
                    if param.shape[0] != reconstructed_params[name].shape[0]:
                        continue
                skip_this_param = False
                if 'none' not in skip_list:  
                    if 'qkv' in skip_list and 'qkv.' in name:
                        skip_this_param = True
                    elif 'o' in skip_list and 'attn.proj.' in name: 
                        skip_this_param = True
                    elif 'mlp' in skip_list and '.mlp.' in name:
                        skip_this_param = True
                    elif 'norm' in skip_list and 'norm' in name:
                        skip_this_param = True

                if skip_this_param:
                    continue
                param.copy_(reconstructed_params[name])
                print(f"[{name}] successfully copied and transferred via frequency domain!")

    return model

# ==========================================
# 2. Encapsulated Initialization Interface
# ==========================================
def init_freq_domain_transfer(model_base, model, args):
    """
    Given model_base and model, extract frequency domain features directly in memory 
    and initialize the target model.
    """
    print(f"==> Start extracting frequency domain representation from {args.basemodel} (Mode: {args.mlp_dct_dim.upper()})...")
    # Pass mlp_dct_dim to control 1D/2D
    todct = ToDCT(keep_ratio=args.keep_ratio, model_name=f'dct_{args.basemodel}_pretrained', mlp_dct_dim=args.mlp_dct_dim)
    
    # Complete feature extraction in memory
    layer_blocks_freq, layer_blocks_norm, layer_presevered_freq = todct(model_base)
    
    print("==> Injecting frequency knowledge into the target model...")
    # Reconstruct and assign to the target model in memory
    model = reconstruct_weights(
        model,
        layer_blocks_freq,
        layer_blocks_norm,
        layer_presevered_freq,
        use_dct=args.use_dct,
        use_ratio=args.use_ratio,
        select_n=args.select_n,
        mode=args.dct_mode,
        skip_list=args.dct_skip_list,
        mlp_dct_dim=args.mlp_dct_dim  # Pass the IDCT dimension mode
    )
    print("==> Frequency domain knowledge transfer complete!")
    return model

# ==========================================
# 3. Main Function and Execution Entry
# ==========================================
def get_args_parser():
    parser = argparse.ArgumentParser('Frequency Domain Knowledge Transfer Testing', add_help=False)
    # Simulate the args parameters from your previous code
    parser.add_argument('--dct', action='store_true', help='Enable DCT transfer')
    parser.add_argument('--pretrained', default=False, type=bool, help='Use pretrained model')
    
    parser.add_argument('--basemodel', default='deit_tiny_patch16_224_L12', type=str, help='Base model architecture')
    parser.add_argument('--model', default='deit_tiny_patch16_224_L4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--basepretrained', action='store_true', help='Use pretrained base model')
    parser.add_argument('--basenb_classes', default=1000, type=int, help='Base model classes')
    parser.add_argument('--nb_classes', default=1000, type=int, help='Model classes')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Drop rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--gene_config', type=str, default=None, help='Genotype config if using wt')
    parser.add_argument('--weight_init', type=str, default='', help='Weight initialization mode')
    parser.add_argument('--basemodel_pretrain_pth', type=str, default='', help='Path to pretrain weights')
    
    # DCT specific args
    parser.add_argument('--use_dct', action='store_true', default=True, help='Use DCT over stacking layers')
    parser.add_argument('--use_ratio', action='store_true', default=False, help='Use area ratio for reconstruction')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='how much frequency components to select')
    parser.add_argument('--select_n', type=int, default=None, help='Top N selected modules')
    parser.add_argument('--dct_mode', type=str, default='zero', choices=['zero', 'random', 'consistence', 'repeat'])
    parser.add_argument('--dct_skip_list', nargs='+', default=['none'], help='Modules to skip during copy')

    
    # ========== New: Control 1D or 2D frequency domain transfer ==========
    parser.add_argument('--mlp_dct_dim', default='2d', type=str, choices=['1d', '2d'], 
                        help='Whether to use 1d or 2d DCT for MLP hidden dimension knowledge transfer')
    
    parser.add_argument('--output_dir', default='', help='save checkpoint to the specified dir')
    
    return parser



def main(args):
    # [Note] Here torchvision / timm are used as external dependency examples, 
    # please replace with your project's create_model in practice
    try:
        from timm import create_model
    except ImportError:
        print("Warning: 'timm' not found. Please ensure your custom 'create_model' is accessible.")
        # Place your create_model logic here

    # Simulate the instantiation of the target small model (please replace with your actual target model)
    print(f"Creating Target Model (to be initialized)...")
    if 'wt' in args.model:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            gene_config=args.gene_config
        )
    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            weight_init = args.weight_init
        )

    if args.dct:
        print(f"Creating Base Model: {args.basemodel}...")
        # 1. Initialize and load Base Model parameters
        if 'wt' in args.basemodel:
            model_base = create_model(
                args.basemodel,
                pretrained=args.basepretrained,
                num_classes=args.basenb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                gene_config=args.gene_config
            )
        else:
            model_base = create_model(
                args.basemodel,
                pretrained=args.basepretrained,
                num_classes=args.basenb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                weight_init=args.weight_init
            )
            
        print("Model Base loaded successfully.")
        
        # Handle pretrained weights loading logic
        model_path = args.basemodel_pretrain_pth
        if model_path:
            print(f"Loading weights from {model_path}...")
            if model_path.endswith('.npz'):
                model_base.load_pretrained(model_path)
            elif model_path.endswith('.pth'):
                model_data = torch.load(model_path, map_location='cpu')
                if 'model' in model_data:
                    model_base.load_state_dict(model_data['model'])
                else:
                    model_base.load_state_dict(model_data)
            else:
                raise ValueError("Unsupported model file extension. Only .npz or .pth are supported.")

        # 2. Initialize Target Model directly via memory
        model = init_freq_domain_transfer(model_base, model, args)
        
        keep_ratio_tag = f"_keepwidth_{args.keep_ratio}" if args.keep_ratio != 1.0 else ""
        save_name = f'/from_{args.basemodel}_to_{args.model}_init_mlp_{args.mlp_dct_dim}{keep_ratio_tag}.pth'
        torch.save(model.state_dict(), args.output_dir + save_name)
        
        print("\n[Success] Model is ready for training/finetuning.")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # If you run tests directly in an IDE, you can manually set the required args like below:
    args.dct = True
    args.model = 'your target model name'
    args.basemodel = 'your base model name' 
    args.basepretrained = False
    args.select_n = None 
    args.keep_ratio = 1.0
    args.mlp_dct_dim = '2d'  # Switch between '1d' or '2d' here
    args.basemodel_pretrain_pth = 'your_base_model_pretrained.pth'
    args.output_dir = 'your_output_dir_path'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)