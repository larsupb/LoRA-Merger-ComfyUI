#
# File from: https://raw.githubusercontent.com/mgz-dev/sd-scripts/main/networks/resize_lora.py
#

# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo and kohya

import torch
import numpy as np

import comfy.utils
import comfy.model_management
from comfy.weight_adapter import LoRAAdapter
from .architectures.sd_lora import weights_as_tuple
from .utility import index_sv_cumulative, index_sv_fro, find_network_dim, perform_lora_svd, perform_lora_qr

MIN_SV = 1e-6


def find_network_dim_from_adapter(adapter_dict):
    """
    Find the network dimension (rank) from a dictionary of LoRAAdapter objects.

    Args:
        adapter_dict: Dict[str, LoRAAdapter]

    Returns:
        int: The rank of the LoRA
    """
    for adapter in adapter_dict.values():
        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        if down is not None and hasattr(down, 'size'):
            # For 2D tensors, rank is down.size()[0]
            # For 4D conv tensors, rank is down.size()[0]
            return down.size()[0]
    return None


def lora_adapter_to_state_dict(adapter_dict):
    """
    Convert a dictionary of LoRAAdapter objects to a raw state dict with tensors.

    This is similar to architectures.sd_lora.convert_to_regular_lora() but simpler:
    - convert_to_regular_lora(): SD keys → diffusers keys (for saving to disk, requires model/clip)
    - lora_adapter_to_state_dict(): LoRAAdapter → raw tensors (for internal processing, no key remapping)

    Args:
        adapter_dict: Dict[str, LoRAAdapter] - Dictionary mapping keys to LoRAAdapter objects

    Returns:
        dict: State dict with tensor values in the format expected by resize_lora_model
    """
    state_dict = {}
    for key, adapter in adapter_dict.items():
        if isinstance(adapter, LoRAAdapter):
            # Extract weights from LoRAAdapter: (mat1/up, mat2/down, alpha, mid, dora_scale, reshape)
            mat1, mat2, alpha, mid, dora_scale, reshape = adapter.weights

            # The key in adapter_dict is the base layer name (e.g., "diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_k")
            # We need to add the standard LoRA suffixes for resize_lora_model
            state_dict[f"{key}.lora_up.weight"] = mat1
            state_dict[f"{key}.lora_down.weight"] = mat2

            # Handle alpha which can be None, a tensor, or a scalar
            if alpha is None:
                alpha = 1.0  # Default alpha value
            state_dict[f"{key}.alpha"] = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)

            if mid is not None:
                state_dict[f"{key}.lora_mid.weight"] = mid
        else:
            # If it's already a raw tensor, keep it as-is (for backward compatibility)
            state_dict[key] = adapter

    return state_dict


def state_dict_to_lora_adapter(state_dict, loaded_keys=None):
    """
    Convert a raw state dict back to a dictionary of LoRAAdapter objects.

    Args:
        state_dict: dict - State dict with tensor values
        loaded_keys: set - Set of loaded keys for LoRAAdapter

    Returns:
        Dict[str, LoRAAdapter]: Dictionary mapping keys to LoRAAdapter objects
    """
    if loaded_keys is None:
        loaded_keys = set(state_dict.keys())

    adapter_dict = {}
    processed_keys = set()

    for key in state_dict.keys():
        if key in processed_keys:
            continue

        # Look for lora_up/lora_down pairs
        if '.lora_up.weight' in key:
            base_key = key.replace('.lora_up.weight', '')
            up_key = f"{base_key}.lora_up.weight"
            down_key = f"{base_key}.lora_down.weight"
            alpha_key = f"{base_key}.alpha"
            mid_key = f"{base_key}.lora_mid.weight"

            if up_key in state_dict and down_key in state_dict:
                up = state_dict[up_key]
                down = state_dict[down_key]
                alpha = state_dict.get(alpha_key, torch.tensor(1.0))
                mid = state_dict.get(mid_key, None)

                # Create LoRAAdapter with weights tuple
                # Use weights_as_tuple for consistency with the rest of the codebase
                # Note: weights_as_tuple creates (up, down, alpha, None, None, None)
                # We extend it to support mid if present
                # Alpha can be a tensor or scalar - both are valid in the LoRA format
                if mid is not None:
                    weights = (up, down, alpha, mid, None, None)
                else:
                    weights = weights_as_tuple(up, down, alpha)
                adapter_dict[base_key] = LoRAAdapter(loaded_keys=loaded_keys, weights=weights)

                processed_keys.add(up_key)
                processed_keys.add(down_key)
                processed_keys.add(alpha_key)
                if mid_key in state_dict:
                    processed_keys.add(mid_key)

    return adapter_dict


class LoraResizer:
    def __init__(self):
        self.loaded_lora = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LoRABundle",),
                "new_rank": ("INT", {
                    "default": 16,
                    "min": 1,  # Minimum value
                    "max": 320,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "method": (["SVD", "QR"],),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRABundle",)
    FUNCTION = "lora_svd_resize"
    CATEGORY = "LoRA PowerMerge"

    def lora_svd_resize(self, lora, new_rank=None, method="SVD", device=None, dtype=None):
        """
            Resize the given LoRA model to a new rank using SVD or QR decomposition.

            This method adjusts the rank of the LoRA model's state dictionary by performing
            a decomposition-based resizing operation. If the current rank matches the new rank,
            no resizing is performed. The method ensures all computations are done in `float32`
            to avoid precision issues.

            Args:
                lora (dict): A dictionary containing the LoRA model. The model's state dictionary should
                    be accessible via the 'lora' key (from merger) or 'lora_raw' key (from loader).
                new_rank (int, optional): The desired new rank for the LoRA model. If `None`, no resizing
                    will occur unless the current rank is different from `new_rank`.
                method (str): Decomposition method to use - "SVD" (slower, more accurate) or
                    "QR" (faster, less optimal). Default: "SVD".
                device (torch.DeviceObjType, optional): The device on which to perform the computations
                    (e.g., 'cuda' or 'cpu'). If `None`, the default device will be used.
                dtype (torch.dtype, optional): The data type for the resized tensor. If `None`, the
                    original data type will be retained.

            Returns:
                tuple: A tuple containing the resized LoRA model.

            Note:
                - The method internally converts tensors to `float32` for the resizing process, ensuring
                  compatibility with the decomposition operations.
                - The resizing is skipped if the current rank matches the specified `new_rank`.
                - QR is typically 2-5x faster than SVD but may result in slightly lower quality approximation.
        """
        # Add fallback for lora_raw format (from PM LoRA Loader)
        # This matches the pattern used in lora_apply.py for compatibility
        if 'lora' not in lora:
            if 'lora_raw' in lora:
                # Use raw state dict directly (contains tensor keys like "layer.lora_up.weight")
                lora['lora'] = lora['lora_raw']
            else:
                raise ValueError("LoRA data missing both 'lora' and 'lora_raw' keys")

        state_dict = lora['lora']

        # Check if state_dict contains LoRAAdapter objects (from merger) or raw tensors (from loader)
        first_value = next(iter(state_dict.values()))
        is_lora_adapter_format = hasattr(first_value, 'weights')  # LoRAAdapter has .weights attribute

        if is_lora_adapter_format:
            # LoRAAdapter format from merger
            current_rank = find_network_dim_from_adapter(state_dict)

            # If rank already matches, pass through unchanged
            if current_rank == new_rank:
                return (lora,)

            # Resizing merged LoRAs (LoRAAdapter format) is not fully supported
            # The recommended workflow is: Resize individual LoRAs BEFORE merging
            #
            # However, we'll attempt to resize by converting to raw format and back
            # This may result in "NOT LOADED" warnings if the conversion loses metadata

            print(f"PM LoRA Resizer: Warning - Resizing merged LoRA from rank {current_rank} to {new_rank}.")
            print(f"PM LoRA Resizer: For best results, resize individual LoRAs before merging.")

            # Save the merged LoRA to a temporary state dict, resize it, then reload
            # This is a workaround - the proper solution is to resize before merging
            raise NotImplementedError(
                "Resizing merged LoRAs (from PM LoRA Merger) is not currently supported. "
                "Please use this workflow instead:\n"
                "1. Load each LoRA individually\n"
                "2. Resize each LoRA to the desired rank using PM LoRA Resizer\n"
                "3. Then merge the resized LoRAs using PM LoRA Merger\n\n"
                "Alternatively, save the merged LoRA, then load and resize it."
            )
        else:
            # Raw tensor format from loader - process directly
            if find_network_dim(state_dict) != new_rank:
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        state_dict[key] = value.to(dtype=torch.float32)

                resized, _, _ = resize_lora_model(lora_sd=state_dict, new_rank=new_rank, save_dtype=dtype, device=device,
                                                  dynamic_method=None, dynamic_param=None, verbose=False, method=method)
                state_dict = resized

            lora['lora'] = state_dict

        return (lora,)


def resize_lora_model(lora_sd, new_rank, save_dtype, device, dynamic_method, dynamic_param, verbose, method="SVD"):
    network_alpha = None
    network_dim = None
    verbose_str = "\n"
    fro_list = []
    save_dtype = str_to_dtype(save_dtype)

    # Extract loaded lora dim and alpha
    for key, value in lora_sd.items():
        if network_alpha is None and 'alpha' in key:
            network_alpha = value
        if network_dim is None and 'lora_down' in key:
            # Check if value is a tensor with the expected shape
            if hasattr(value, 'size') and len(value.size()) == 2:
                network_dim = value.size()[0]
        if network_alpha is not None and network_dim is not None:
            break

    # Fallback: if alpha not found, use network_dim
    if network_alpha is None:
        network_alpha = network_dim

    # Safety check to prevent division by None
    if network_alpha is None or network_dim is None:
        # Provide detailed error with sample keys to help debugging
        sample_keys = list(lora_sd.keys())[:5]
        raise ValueError(f"Could not determine LoRA dimensions. Found alpha={network_alpha}, dim={network_dim}. "
                        f"Sample keys in state dict: {sample_keys}. "
                        f"Please ensure the LoRA file has valid 'lora_down' and 'alpha' keys.")

    scale = network_alpha / network_dim

    if dynamic_method:
        print(
            f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}")

    lora_down_weight = None
    lora_up_weight = None

    o_lora_sd = lora_sd.copy()
    block_down_name = None
    block_up_name = None

    # Create ComfyUI progress bar for better UX and interrupt support
    pbar = comfy.utils.ProgressBar(len(lora_sd))

    with torch.no_grad():
        for key, value in lora_sd.items():
            # Check for interrupt signal from ComfyUI
            comfy.model_management.throw_exception_if_processing_interrupted()

            # Update progress bar
            pbar.update(1)

            if 'lora_down' in key:
                block_down_name = key.split(".")[0]
                lora_down_weight = value
            if 'lora_up' in key:
                block_up_name = key.split(".")[0]
                lora_up_weight = value

            weights_loaded = (lora_down_weight is not None and lora_up_weight is not None)

            if (block_down_name == block_up_name) and weights_loaded:

                conv2d = (len(lora_down_weight.size()) == 4)

                if conv2d:
                    full_weight_matrix = merge_conv(lora_down_weight, lora_up_weight, device)
                    param_dict = extract_conv(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device,
                                              scale, method)
                else:
                    full_weight_matrix = merge_linear(lora_down_weight, lora_up_weight, device)
                    param_dict = extract_linear(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device,
                                                scale, method)

                if verbose:
                    max_ratio = param_dict['max_ratio']
                    sum_retained = param_dict['sum_retained']
                    fro_retained = param_dict['fro_retained']
                    if not np.isnan(fro_retained):
                        fro_list.append(float(fro_retained))

                    verbose_str += f"{block_down_name:75} | "
                    verbose_str += f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"

                if verbose and dynamic_method:
                    verbose_str += f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}\n"
                else:
                    verbose_str += f"\n"

                new_alpha = param_dict['new_alpha']
                o_lora_sd[block_down_name + "." + "lora_down.weight"] = param_dict["lora_down"].to(
                    save_dtype).contiguous()
                o_lora_sd[block_up_name + "." + "lora_up.weight"] = param_dict["lora_up"].to(save_dtype).contiguous()
                o_lora_sd[block_up_name + "." "alpha"] = torch.tensor(param_dict['new_alpha']).to(save_dtype)

                block_down_name = None
                block_up_name = None
                lora_down_weight = None
                lora_up_weight = None
                weights_loaded = False
                del param_dict

    if verbose:
        print(verbose_str)

        print(f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}")
    print("resizing complete")
    return o_lora_sd, network_dim, new_alpha


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1, method="SVD"):
    """
    Extract LoRA from conv layer using SVD or QR decomposition.
    """
    # Choose decomposition method
    if method == "QR":
        # QR doesn't support dynamic methods, use fixed rank
        if dynamic_method is not None:
            print(f"Warning: QR decomposition doesn't support dynamic rank selection ({dynamic_method}). Using fixed rank {lora_rank}.")
        up, down, new_alpha, stats = perform_lora_qr(
            weight=weight,
            target_rank=lora_rank,
            device=device,
            scale=scale,
            distribute_singular_values=False,  # Asymmetric: all values in up
            return_statistics=True
        )
    else:  # SVD
        # Use unified SVD function with asymmetric distribution and statistics
        up, down, new_alpha, stats = perform_lora_svd(
            weight=weight,
            target_rank=lora_rank,
            device=device,
            dynamic_method=dynamic_method,
            dynamic_param=dynamic_param,
            scale=scale,
            distribute_singular_values=False,  # Asymmetric: all S in up
            return_statistics=True
        )

    # Build result dictionary matching original format
    param_dict = {
        "lora_up": up,
        "lora_down": down,
        "new_rank": stats['new_rank'],
        "new_alpha": stats['new_alpha'],
        "sum_retained": stats['sum_retained'],
        "fro_retained": stats['fro_retained'],
        "max_ratio": stats['max_ratio']
    }

    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1, method="SVD"):
    """
    Extract LoRA from linear layer using SVD or QR decomposition.
    """
    # Choose decomposition method
    if method == "QR":
        # QR doesn't support dynamic methods, use fixed rank
        if dynamic_method is not None:
            print(f"Warning: QR decomposition doesn't support dynamic rank selection ({dynamic_method}). Using fixed rank {lora_rank}.")
        up, down, new_alpha, stats = perform_lora_qr(
            weight=weight,
            target_rank=lora_rank,
            device=device,
            scale=scale,
            distribute_singular_values=False,  # Asymmetric: all values in up
            return_statistics=True
        )
    else:  # SVD
        # Use unified SVD function with asymmetric distribution and statistics
        up, down, new_alpha, stats = perform_lora_svd(
            weight=weight,
            target_rank=lora_rank,
            device=device,
            dynamic_method=dynamic_method,
            dynamic_param=dynamic_param,
            scale=scale,
            distribute_singular_values=False,  # Asymmetric: all S in up
            return_statistics=True
        )

    # Build result dictionary matching original format
    param_dict = {
        "lora_up": up,
        "lora_down": down,
        "new_rank": stats['new_rank'],
        "new_alpha": stats['new_alpha'],
        "sum_retained": stats['sum_retained'],
        "fro_retained": stats['fro_retained'],
        "max_ratio": stats['max_ratio']
    }

    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank and kernel_size == k_, f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight


def str_to_dtype(p):
    if p == 'float':
        return torch.float
    if p == 'fp16':
        return torch.float16
    if p == 'bf16':
        return torch.bfloat16
    return None
