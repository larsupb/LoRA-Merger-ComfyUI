"""
Direct LoRA loading from safetensors files.

No ComfyUI dependencies - pure safetensors + torch.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from .key_utils import build_normalized_index, NormalizedKey, normalize_diffusers_key, normalize_kohya_key

logger = logging.getLogger(__name__)


# src/merge/lora_loader.py

def load_lora_safetensors(
        path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """
    Load a LoRA from safetensors file.

    Preserves original key format for later saving.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA not found: {path}")

    logger.info(f"Loading LoRA: {path}")

    # Load raw tensors
    tensors = load_file(str(path))

    if device or dtype:
        tensors = {
            k: v.to(device=device, dtype=dtype) if dtype else v.to(device=device)
            for k, v in tensors.items()
        }

    # Separate by weight type and build normalized index
    up_tensors: Dict[NormalizedKey, torch.Tensor] = {}
    down_tensors: Dict[NormalizedKey, torch.Tensor] = {}
    alpha_tensors: Dict[NormalizedKey, torch.Tensor] = {}

    # Keep original keys for saving later
    original_keys: Dict[NormalizedKey, Dict[str, str]] = {}  # {norm_key: {"up": orig, "down": orig, "alpha": orig}}

    for orig_key, tensor in tensors.items():
        # Detect format and normalize
        if orig_key.startswith("lora_unet_"):
            norm_key = normalize_kohya_key(orig_key)
        else:
            norm_key = normalize_diffusers_key(orig_key)

        if norm_key is None:
            logger.debug(f"Could not normalize key: {orig_key}")
            continue

        # Determine weight type and store
        key_lower = orig_key.lower()

        if norm_key not in original_keys:
            original_keys[norm_key] = {}

        if ".lora_up" in key_lower or ".lora_b" in key_lower:
            up_tensors[norm_key] = tensor
            original_keys[norm_key]["up"] = orig_key
        elif ".lora_down" in key_lower or ".lora_a" in key_lower:
            down_tensors[norm_key] = tensor
            original_keys[norm_key]["down"] = orig_key
        elif ".alpha" in key_lower:
            alpha_tensors[norm_key] = tensor
            original_keys[norm_key]["alpha"] = orig_key

    logger.info(f"Loaded {len(original_keys)} normalized layers from {path.name}")

    return {
        "tensors": tensors,
        "up_tensors": up_tensors,
        "down_tensors": down_tensors,
        "alpha_tensors": alpha_tensors,
        "original_keys": original_keys,  # Preserve for saving
        "path": str(path),
        "name": path.stem,
    }


def save_lora_safetensors(
        up_tensors: Dict[NormalizedKey, torch.Tensor],
        down_tensors: Dict[NormalizedKey, torch.Tensor],
        alpha_tensors: Dict[NormalizedKey, torch.Tensor],
        original_keys: Dict[NormalizedKey, Dict[str, str]],  # Use original key templates
        output_path: str,
        metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save merged LoRA tensors using original key format.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_tensors = {}

    for norm_key, tensor in down_tensors.items():
        if norm_key in original_keys and "down" in original_keys[norm_key]:
            # Use original key
            orig_key = original_keys[norm_key]["down"]
            save_tensors[orig_key] = tensor.cpu().contiguous()
        else:
            # Fallback to generated key (shouldn't happen often)
            logger.warning(f"No original key for {norm_key}, generating fallback")
            fallback_key = _normalized_to_kohya_sd_native(norm_key, "down")
            save_tensors[fallback_key] = tensor.cpu().contiguous()

    for norm_key, tensor in up_tensors.items():
        if norm_key in original_keys and "up" in original_keys[norm_key]:
            orig_key = original_keys[norm_key]["up"]
            save_tensors[orig_key] = tensor.cpu().contiguous()
        else:
            fallback_key = _normalized_to_kohya_sd_native(norm_key, "up")
            save_tensors[fallback_key] = tensor.cpu().contiguous()

    for norm_key, tensor in alpha_tensors.items():
        if norm_key in original_keys and "alpha" in original_keys[norm_key]:
            orig_key = original_keys[norm_key]["alpha"]
            save_tensors[orig_key] = tensor.cpu().contiguous()
        else:
            fallback_key = _normalized_to_kohya_sd_native(norm_key, "alpha")
            save_tensors[fallback_key] = tensor.cpu().contiguous()

    save_file(save_tensors, str(output_path), metadata=metadata or {})
    logger.info(f"Saved merged LoRA: {output_path} ({len(save_tensors)} keys)")

    return str(output_path)


def _normalized_to_kohya_sd_native(norm_key: NormalizedKey, weight_type: str) -> str:
    """
    Convert NormalizedKey to Kohya SD-native format (fallback).

    SD1.5 UNet structure:
    - input_blocks: 0=stem, 1-3=down_block_0, 4-6=down_block_1, 7-9=down_block_2, 10-11=down_block_3
    - Within each group: 0=resnet, 1=attention (or resnet), 2=attention (or downsample)
    - middle_block: 0=resnet, 1=attention, 2=resnet
    - output_blocks: similar structure in reverse

    """
    parts = ["lora_unet"]

    if norm_key.block_type == "down":
        # Convert diffusers down_blocks.X.attentions.Y to input_blocks.N.M
        # input_blocks index = 1 + block_idx*3 + (1 if first attention, 2 if second)
        # This is approximate - the exact mapping depends on architecture
        base_idx = 1 + norm_key.block_idx * 3
        sub_idx = 1 + norm_key.attention_idx  # attention layers are at sub-index 1 or 2
        parts.append(f"input_blocks_{base_idx + norm_key.attention_idx}_1")

    elif norm_key.block_type == "mid":
        parts.append("middle_block_1")  # attention is at index 1

    elif norm_key.block_type == "up":
        # Similar logic for output_blocks
        base_idx = norm_key.block_idx * 3
        parts.append(f"output_blocks_{base_idx + norm_key.attention_idx}_1")

    # Transformer block
    parts.append(f"transformer_blocks_{norm_key.transformer_idx}")

    # Attention type
    if norm_key.attn_type == "self":
        parts.append("attn1")
    elif norm_key.attn_type == "cross":
        parts.append("attn2")

    # Sublayer - handle special cases
    sublayer = norm_key.sublayer
    if sublayer == "to_out":
        sublayer = "to_out_0"  # ModuleList index
    elif sublayer == "ff.net.0.proj":
        sublayer = "ff_net_0_proj"
    elif sublayer == "ff.net.2":
        sublayer = "ff_net_2"
    else:
        sublayer = sublayer.replace(".", "_")

    parts.append(sublayer)

    # Join with underscores
    base = "_".join(parts)

    # Add weight type suffix
    if weight_type == "down":
        return f"{base}.lora_down.weight"
    elif weight_type == "up":
        return f"{base}.lora_up.weight"
    else:
        return f"{base}.alpha"