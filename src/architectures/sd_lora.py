import logging
import math
import re
from typing import List, Set, Dict

import comfy
import torch
from comfy.model_patcher import ModelPatcher
from torch import Tensor

from ..types import (
    LORA_STACK,
    LORA_KEY_DICT,
    LORA_TENSOR_DICT,
    BlockNameInfo,
)


def weights_as_tuple(up: torch.Tensor, down: torch.Tensor, alpha: torch.Tensor):
    return (up, down, alpha, None, None, None)


def analyse_keys(loras: LORA_STACK) -> Set[str]:
    keys = set()
    for i, lora in enumerate(loras.values()):
        key_count = 0
        for key in lora.keys():
            keys.add(key)
            key_count += 1
        logging.debug(f"LoRA {i} with {key_count} modules.")

    logging.info(f"Total keys to be merged: {len(keys)} modules")
    return keys


def calc_up_down_alphas(
    loras_lora_key_dict: LORA_STACK,
    key: str,
    load_device: torch.device = None,
    scale_to_alpha_0: bool = False
) -> LORA_TENSOR_DICT:
    """
       Calculate up, down tensors and alphas for a given key.

       Args:
           loras_lora_key_dict: Dictionary containing LoRA names and their respective keys.
           key: The key to calculate values for.
           load_device: Device to load tensors on.
           scale_to_alpha_0: Whether to scale alphas to the alpha of lora 0.

       Returns:
           List of tuples containing up, down tensors and alpha values.
    """
    owners = [lora_name for lora_name, lora_key_dict in loras_lora_key_dict.items() if key in lora_key_dict]
    if len(owners) == 0:
        raise ValueError(f"Key {key} not found in any LoRA provided.")

    # Define tensor index positions
    down_idx, up_idx, alpha_idx = (1, 0, 2)

    # Determine alpha from the first lora which contains the module
    owner_0_key_dict = loras_lora_key_dict[owners[0]]
    alpha_0 = owner_0_key_dict[key].weights[alpha_idx]

    out = {}
    for lora_name in owners:
        lora_key_dict = loras_lora_key_dict[lora_name]
        lora_tensors = lora_key_dict[key].weights
        up = lora_tensors[up_idx]
        down = lora_tensors[down_idx]
        alpha = lora_tensors[alpha_idx]

        if scale_to_alpha_0 and alpha_0 is not None and alpha != alpha_0:
            up = up * math.sqrt(alpha / alpha_0)
            down = down * math.sqrt(alpha / alpha_0)
            alpha = alpha_0

        if load_device is not None:
            up = up.to(device=load_device, dtype=torch.float32)
            down = down.to(device=load_device, dtype=torch.float32)
        up_down_alpha = up, down, alpha
        out[lora_name] = up_down_alpha

    return out


@DeprecationWarning
def scale_alphas(ups_downs_alphas):
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = []
    for up, down, alpha in ups_downs_alphas:
        up = up * math.sqrt(alpha / alpha_1)
        down = down * math.sqrt(alpha / alpha_1)
        out.append((up, down, alpha_1))
    return out, alpha_1


def sd_to_diffusers_map(model: ModelPatcher) -> Dict[str, str]:
    diffusers_keys = comfy.utils.unet_to_diffusers(model.model.model_config.unet_config)

    key_map = {}
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key  # simpletuner lycoris format

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    # Create inverse entries for key_map
    inverse_map = {v: k for k, v in key_map.items()}
    return inverse_map


def convert_to_regular_lora(model, state_dict: LORA_KEY_DICT):
    diffusers_map = sd_to_diffusers_map(model)
    out = {}

    # Group split QKV weights by base key
    qkv_groups = {}
    non_qkv_keys = []

    for sd_key in state_dict.keys():
        # Check if this is a split QKV key (tuple format with offset)
        if isinstance(sd_key, tuple) and len(sd_key) == 2:
            base_key = sd_key[0]
            if ".qkv.weight" in base_key or ".in_proj_weight" in base_key:
                if base_key not in qkv_groups:
                    qkv_groups[base_key] = []
                qkv_groups[base_key].append(sd_key)
            else:
                non_qkv_keys.append(sd_key)
        else:
            non_qkv_keys.append(sd_key)

    # Process QKV groups - concatenate split components
    for base_key, split_keys in qkv_groups.items():
        # Sort by offset to ensure correct order (Q, K, V)
        split_keys_sorted = sorted(split_keys, key=lambda x: x[1][1])  # Sort by offset

        # Extract tensors from each split
        up_parts = []
        down_parts = []
        alphas = []

        for sk in split_keys_sorted:
            lora_settings = state_dict[sk]
            up, down, alpha, mid, dora_scale, reshape = lora_settings.weights
            up_parts.append(up)
            down_parts.append(down)
            alphas.append(alpha if alpha is not None else 1.0)

        # Check if all alphas are the same
        alpha_val = alphas[0]
        if not all(abs(a - alpha_val) < 1e-6 for a in alphas):
            logging.warning(f"QKV components have different alphas: {alphas}. Using first alpha: {alpha_val}")

        # Concatenate along the output dimension (dim=0 for up tensor)
        up_combined = torch.cat(up_parts, dim=0)
        # down tensors should be the same for all splits, just take the first one
        down_combined = down_parts[0]

        # Generate key for combined QKV
        key_base = base_key.replace("diffusion_model.", "")
        if key_base.endswith(".weight"):
            key_base = key_base[:-len(".weight")]
        key_suffix = key_base.replace(".", "_")

        up_key = "lora_unet_{}.lora_up.weight".format(key_suffix)
        down_key = "lora_unet_{}.lora_down.weight".format(key_suffix)
        alpha_key = "lora_unet_{}.alpha".format(key_suffix)

        # Convert alpha to tensor, handling both float and tensor inputs
        if alpha_val is None:
            alpha_tensor = torch.tensor(1.0)
        elif isinstance(alpha_val, torch.Tensor):
            alpha_tensor = alpha_val.clone().detach().cpu()
        else:
            alpha_tensor = torch.tensor(float(alpha_val))

        out[up_key] = up_combined
        out[down_key] = down_combined
        out[alpha_key] = alpha_tensor

    # Process non-QKV keys normally
    for sd_key in non_qkv_keys:
        lora_settings = state_dict[sd_key]
        lora_type: str = lora_settings.name
        lora_data: List[Tensor] = lora_settings.weights
        if lora_type == "lora":
            # ComfyUI uses tuple keys - convert to string if needed
            key_str = sd_key[0] if isinstance(sd_key, tuple) else sd_key

            # Check if this is a DiT architecture key (doesn't need conversion)
            # DiT keys look like: diffusion_model.layers.X.feed_forward.w1.weight
            if sd_key not in diffusers_map:
                # For DiT architecture, keys are already in correct format
                # Just convert to standard LoRA format with .lora_up/.lora_down suffix
                up, down, alpha, mid, dora_scale, reshape = lora_data

                # Strip .weight suffix if present before converting to LoRA format
                # ComfyUI internal keys end with .weight, but LoRA keys should not have it in the middle
                key_base = key_str.replace("diffusion_model.", "")
                if key_base.endswith(".weight"):
                    key_base = key_base[:-len(".weight")]

                # Convert dots to underscores for LoRA naming convention
                key_suffix = key_base.replace(".", "_")
                up_key = "lora_unet_{}.lora_up.weight".format(key_suffix)
                down_key = "lora_unet_{}.lora_down.weight".format(key_suffix)
                alpha_key = "lora_unet_{}.alpha".format(key_suffix)

                # Convert alpha to tensor, handling both float and tensor inputs
                if alpha is None:
                    alpha_tensor = torch.tensor(1.0)
                elif isinstance(alpha, torch.Tensor):
                    alpha_tensor = alpha.clone().detach().cpu()
                else:
                    alpha_tensor = torch.tensor(float(alpha))

                out[up_key] = up
                out[down_key] = down
                out[alpha_key] = alpha_tensor
            else:
                # Standard SD UNet architecture - use mapping
                out_key = diffusers_map[sd_key]

                up, down, alpha, mid, dora_scale, reshape = lora_data
                up_key = "{}.lora_up.weight".format(out_key)
                down_key = "{}.lora_down.weight".format(out_key)
                alpha_key = "{}.alpha".format(out_key)

                # Convert alpha to tensor, handling both float and tensor inputs
                if alpha is None:
                    alpha_tensor = torch.tensor(1.0)
                elif isinstance(alpha, torch.Tensor):
                    alpha_tensor = alpha.clone().detach().cpu()
                else:
                    alpha_tensor = torch.tensor(float(alpha))

                out[up_key] = up
                out[down_key] = down
                out[alpha_key] = alpha_tensor
        else:
            raise ValueError(f"Currently only LoRA type: {lora_type} is supported.")
    return out


def detect_block_names(layer_key: str) -> BlockNameInfo:
    # Convert tuple keys to strings (ComfyUI uses tuple keys)
    if isinstance(layer_key, tuple):
        layer_key = layer_key[0]

    # With transformer_blocks
    exp_with_transformer = re.compile(r"""
        (?:diffusion_model\.)?                                  # optional prefix
        (?P<block_type>input_blocks|middle_block|output_blocks)
        \.
        (?P<block_idx>\d+)
        (?:\.(?P<inner_idx>\d+))?                                # optional inner block
        \.
        transformer_blocks
        \.
        (?P<transformer_idx>\d+)
        \.
        (?P<component>attn1|attn2|ff|proj_in|proj_out)           # top-level component
        (?:\..+)?                                                # allow nested submodules (e.g. .to_q.weight)
    """, re.VERBOSE)
    # Projection in and out blocks are not part of transformer_blocks, so we need a simpler regex for those.
    exp_simple = re.compile(r"""
        (?P<block_type>input_blocks|middle_block|output_blocks)
        \.
        (?P<block_idx>\d+)(?:\.(?P<inner_idx>\d+))?
        \.
        (?P<component>proj_in|proj_out)
        (?:\..+)?                                                # allow nested submodules (e.g. .to_q.weight)
    """, re.VERBOSE)

    for exp in [exp_with_transformer, exp_simple]:
        match = exp.search(layer_key)
        if match:
            out = {
                "block_type": match.group("block_type"),
                "block_idx": match.group("block_idx"),
                "inner_idx": match.group("inner_idx"),
                "component": match.group("component"),
                "main_block": f"{match.group("block_type")}.{match.group("block_idx")}",
                "sub_block": f'{match.group("block_type")}.{match.group("block_idx")}',
                "transformer_idx": None
            }
            if "transformer_idx" in match.groupdict():
                out["transformer_idx"] = match.group("transformer_idx")
                out["sub_block"] = f'{out["block_type"]}.{out["block_idx"]}.{out["transformer_idx"]}'
            return out
    return None