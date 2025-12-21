import math
import re
from typing import List, Tuple, Dict, Set

from torch import Tensor

import comfy
import torch

from comfy.model_patcher import ModelPatcher
from .general_architecture import LORA_STACK, LORA_KEY_DICT

SD_LORA_ALPHA_IDX = 2
LORA_TENSORS = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
LORA_TENSOR_DICT = Dict[str, LORA_TENSORS]
LORA_TENSORS_BY_LAYER = Dict[str, LORA_TENSOR_DICT]


def weights_as_tuple(up: torch.Tensor, down: torch.Tensor, alpha: torch.Tensor):
    return (up, down, alpha, None, None, None)


def analyse_keys(loras: LORA_STACK) -> Set[str]:
    keys = set()
    for i, lora in enumerate(loras.values()):
        key_count = 0
        for key in lora.keys():
            keys.add(key)
            key_count += 1
        print(f"LoRA {i} with {key_count} modules.")

    print(f"Total keys to be merged {len(keys)} modules")
    return keys


def calc_up_down_alphas(loras_lora_key_dict: LORA_STACK, key,
                        load_device=None, scale_to_alpha_0=False) -> LORA_TENSOR_DICT:
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
    for sd_key, lora_settings in state_dict.items():
        lora_type: str = lora_settings.name
        lora_data: List[Tensor] = lora_settings.weights
        if lora_type == "lora":
            out_key = diffusers_map[sd_key]

            up, down, alpha, mid, dora_scale, reshape = lora_data
            up_key = "{}.lora_up.weight".format(out_key)
            down_key = "{}.lora_down.weight".format(out_key)
            alpha_key = "{}.alpha".format(out_key)

            # check if alpha is None and set it to 1.0
            if alpha is None:
                alpha = 1.0

            out[up_key] = up
            out[down_key] = down
            out[alpha_key] = torch.tensor(alpha)
        else:
            raise ValueError(f"Currently only LoRA type: {lora_type} is supported.")
    return out


def detect_block_names(layer_key) -> Dict[str, str]:
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