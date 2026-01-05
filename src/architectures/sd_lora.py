import logging
import math
import re
from typing import List, Set, Dict, Tuple, Any

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
    for i, lora_entry in enumerate(loras.values()):
        key_count = 0
        patches = lora_entry["patches"]
        for key in patches.keys():
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
    owners = [lora_name for lora_name, lora_entry in loras_lora_key_dict.items() if key in lora_entry["patches"]]
    if len(owners) == 0:
        raise ValueError(f"Key {key} not found in any LoRA provided.")

    # Log warning if only one owner (helpful for debugging merge failures)
    if len(owners) == 1:
        logging.debug(f"Key '{key}' found in only 1 LoRA: {owners[0]}")

    # Define tensor index positions
    down_idx, up_idx, alpha_idx = (1, 0, 2)

    # Determine alpha from the first lora which contains the module
    owner_0_patches = loras_lora_key_dict[owners[0]]["patches"]
    alpha_0 = owner_0_patches[key].weights[alpha_idx]

    out = {}
    for lora_name in owners:
        patches = loras_lora_key_dict[lora_name]["patches"]
        lora_tensors = patches[key].weights
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


def build_lora_to_unet_key_map(model: ModelPatcher) -> Dict[str, str]:
    """
    Build a forward mapping from LoRA parameter keys to Diffusers UNet weight paths.

    The mapping includes multiple LoRA naming conventions:
    - Standard LoRA safetensors keys:      lora_unet_*
    - LyCORIS / SimpleTuner keys:          lycoris_*
    - Diffusers-native LoRA processor keys (with and without 'unet.' prefix)

    Parameters
    ----------
    model : ModelPatcher
        A ComfyUI ModelPatcher containing a Diffusers-compatible UNet.

    Returns
    -------
    Dict[str, str]
        Mapping from LoRA key name to Diffusers UNet weight path
        (e.g. 'diffusion_model.input_blocks.0.0.transformer_blocks.0.attn1.to_q').
    """
    diffusers_keys = comfy.utils.unet_to_diffusers(
        model.model.model_config.unet_config
    )

    forward_map: Dict[str, str] = {}

    for diffusers_key, sd_key in diffusers_keys.items():
        if not diffusers_key.endswith(".weight"):
            continue

        unet_weight = f"diffusion_model.{sd_key}"
        base_key = diffusers_key[:-len(".weight")]

        # --- ComfyUI / Kohya LoRA format ---
        lora_key = base_key.replace(".", "_")
        forward_map[f"lora_unet_{lora_key}"] = unet_weight
        forward_map[f"lycoris_{lora_key}"] = unet_weight  # SimpleTuner / LyCORIS

        # --- Diffusers native LoRA processor format ---
        for prefix in ("", "unet."):
            diffusers_lora_key = (
                prefix + base_key.replace(".to_", ".processor.to_")
            )

            # diffusers special case: ".to_out.0" → ".to_out"
            if diffusers_lora_key.endswith(".to_out.0"):
                diffusers_lora_key = diffusers_lora_key[:-2]

            forward_map[diffusers_lora_key] = unet_weight

    return forward_map


def invert_lora_to_unet_key_map(
    forward_map: Dict[str, str]
) -> Dict[str, Set[str]]:
    """
    Invert a LoRA→UNet mapping into a UNet→LoRA alias mapping.

    Multiple LoRA key names may legally map to the same UNet weight,
    so the inverse map stores a set of aliases per UNet parameter.

    Parameters
    ----------
    forward_map : Dict[str, str]
        Mapping from LoRA key name to UNet weight path.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping from UNet weight path to a set of LoRA key aliases.
    """
    inverse_map: Dict[str, Set[str]] = {}

    for lora_key, unet_key in forward_map.items():
        inverse_map.setdefault(unet_key, set()).add(lora_key)

    return inverse_map


def convert_to_regular_lora(model, state_dict: LORA_KEY_DICT) -> Dict[str, Tensor]:
    """
    Convert a ComfyUI LoRA state_dict into a regular Kohya/Diffusers-compatible
    LoRA safetensors dictionary.

    Supports:
    - SD / SDXL UNet
    - DiT-style architectures
    - Flux / transformer-based models
    - Split QKV LoRA tensors
    - LyCORIS and DoRA metadata (safely ignored but validated)

    Returns
    -------
    Dict[str, Tensor]
        Flat LoRA state dict with:
        - *.lora_up.weight
        - *.lora_down.weight
        - *.alpha
    """

    # --- Build mappings ---
    forward_map = build_lora_to_unet_key_map(model)
    inverse_map = invert_lora_to_unet_key_map(forward_map)

    out: Dict[str, Tensor] = {}

    # --- Helpers -------------------------------------------------------------

    def normalize_sd_key(sd_key) -> str:
        key = sd_key[0] if isinstance(sd_key, tuple) else sd_key
        if key.endswith(".weight"):
            key = key[:-7]
        return key

    def alpha_to_tensor(alpha) -> Tensor:
        if alpha is None:
            return torch.tensor(1.0)
        if isinstance(alpha, torch.Tensor):
            return alpha.clone().detach().cpu()
        return torch.tensor(float(alpha))

    # --- Split QKV grouping --------------------------------------------------

    qkv_groups: Dict[str, List[Tuple]] = {}
    regular_keys: List[Any] = []

    for sd_key in state_dict.keys():
        if isinstance(sd_key, tuple) and len(sd_key) == 2:
            base = sd_key[0]
            if ".qkv.weight" in base or ".in_proj_weight" in base:
                qkv_groups.setdefault(base, []).append(sd_key)
                continue
        regular_keys.append(sd_key)

    # --- Process QKV LoRA ----------------------------------------------------

    for base_key, split_keys in qkv_groups.items():
        # Sort by offset (Q, K, V)
        split_keys = sorted(split_keys, key=lambda x: x[1][1])

        up_parts, down_parts, alphas = [], [], []
        mids, dora_scales, reshapes = [], [], []

        for sk in split_keys:
            lora_settings = state_dict[sk]
            up, down, alpha, mid, dora_scale, reshape = lora_settings.weights

            up_parts.append(up)
            down_parts.append(down)
            alphas.append(alpha if alpha is not None else 1.0)
            mids.append(mid)
            dora_scales.append(dora_scale)
            reshapes.append(reshape)

        # Safety checks
        if not all(torch.equal(down_parts[0], d) for d in down_parts[1:]):
            raise ValueError(f"Inconsistent down tensors in QKV group: {base_key}")

        if not all(a == alphas[0] for a in alphas):
            logging.warning(
                f"QKV alpha mismatch {alphas} in {base_key}; using first."
            )

        # Combine
        up_combined = torch.cat(up_parts, dim=0)
        down_combined = down_parts[0]
        alpha_tensor = alpha_to_tensor(alphas[0])

        # Key resolution
        norm_key = normalize_sd_key(base_key)

        if norm_key in inverse_map:
            # SD / SDXL UNet path
            lora_base = norm_key
        else:
            # DiT / Flux fallback
            lora_base = f"lora_unet_{norm_key.replace('.', '_')}"

        out[f"{lora_base}.lora_up.weight"] = up_combined
        out[f"{lora_base}.lora_down.weight"] = down_combined
        out[f"{lora_base}.alpha"] = alpha_tensor

    # --- Process regular (non-QKV) LoRA -------------------------------------

    for sd_key in regular_keys:
        lora_settings = state_dict[sd_key]

        if lora_settings.name != "lora":
            raise ValueError(
                f"Unsupported LoRA type: {lora_settings.name}"
            )

        up, down, alpha, mid, dora_scale, reshape = lora_settings.weights
        alpha_tensor = alpha_to_tensor(alpha)

        norm_key = normalize_sd_key(sd_key)

        if norm_key in inverse_map:
            # SD / SDXL UNet
            lora_base = norm_key
        else:
            # DiT / Flux / transformer blocks
            lora_base = f"lora_unet_{norm_key.replace('.', '_')}"

        out[f"{lora_base}.lora_up.weight"] = up
        out[f"{lora_base}.lora_down.weight"] = down
        out[f"{lora_base}.alpha"] = alpha_tensor

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