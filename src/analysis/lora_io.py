"""
LoRA file I/O utilities.

Handles loading and saving LoRA files in various formats.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch
from safetensors.torch import load_file, save_file

from .key_normalizer import LoRAKeyNormalizer, build_base_key_index

logger = logging.getLogger(__name__)


def load_lora(
        path: str,
        device: torch.device = None,
        dtype: torch.dtype = None,
) -> Dict[str, Any]:
    """
    Load a LoRA file and return metadata.

    Args:
        path: Path to .safetensors file
        device: Optional device to load tensors to
        dtype: Optional dtype to cast tensors to

    Returns:
        Dict with:

            - "tensors": Dict[str, torch.Tensor] - The LoRA weights
            - "format": str - Detected format ("kohya", "diffusers_native", etc.)
            - "base_key_index": Dict[str, list[str]] - Base key -> original keys mapping
            - "metadata": Dict - Any metadata from the file

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA file not found: {path}")

    logger.info(f"Loading LoRA from {path}")

    # Load tensors
    tensors = load_file(str(path))

    # Optionally move to device/dtype
    if device is not None or dtype is not None:
        tensors = {
            k: v.to(device=device, dtype=dtype) if dtype else v.to(device=device)
            for k, v in tensors.items()
        }

    # Detect format
    keys = list(tensors.keys())
    format_type = LoRAKeyNormalizer.detect_format(keys)
    logger.info(f"Detected LoRA format: {format_type}")

    # Build base key index for efficient matching
    base_key_index = build_base_key_index(keys, format_type)
    logger.info(f"Built base key index with {len(base_key_index)} unique base keys")

    return {
        "tensors": tensors,
        "format": format_type,
        "base_key_index": base_key_index,
        "path": str(path),
    }


def save_lora(
        tensors: Dict[str, torch.Tensor],
        path: str,
        target_format: str = "kohya",
        metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save LoRA tensors to file.

    Args:
        tensors: Dict of LoRA tensors
        path: Output path (.safetensors)
        target_format: Target format for keys ("kohya", "diffusers_native")
        metadata: Optional metadata to include in file

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to correct format if needed
    source_format = LoRAKeyNormalizer.detect_format(list(tensors.keys()))

    if source_format != target_format and source_format != "unknown":
        logger.info(f"Converting keys from {source_format} to {target_format}")
        tensors = convert_lora_keys(tensors, target_format)

    # Ensure all tensors are on CPU and contiguous for saving
    save_tensors = {}
    for key, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            save_tensors[key] = tensor.cpu().contiguous()
        else:
            save_tensors[key] = torch.tensor(tensor).cpu().contiguous()

    # Save with metadata
    save_file(save_tensors, str(path), metadata=metadata or {})
    logger.info(f"Saved LoRA to {path} ({len(save_tensors)} keys, format: {target_format})")

    return str(path)


def convert_lora_keys(
        tensors: Dict[str, torch.Tensor],
        target_format: str,
) -> Dict[str, torch.Tensor]:
    """
    Convert LoRA keys between formats.

    Currently supports: kohya <-> diffusers_native

    Args:
        tensors: Original tensors dict
        target_format: Target format

    Returns:
        New tensors dict with converted keys
    """
    converted = {}

    for old_key, tensor in tensors.items():
        new_key = _convert_single_key(old_key, target_format)
        converted[new_key] = tensor

    return converted


def _convert_single_key(key: str, target_format: str) -> str:
    """Convert a single key to target format."""
    normalized = LoRAKeyNormalizer.normalize(key)

    if normalized is None:
        logger.warning(f"Could not normalize key, keeping original: {key}")
        return key

    if target_format == "kohya":
        return _normalized_to_kohya(normalized)
    elif target_format == "diffusers_native":
        return _normalized_to_diffusers(normalized)
    else:
        return key


def _normalized_to_kohya(nkey) -> str:
    """Convert NormalizedKey to Kohya format."""
    # Build Kohya-style key
    if nkey.component == "unet":
        parts = ["lora_unet"]
    elif nkey.component == "text_encoder_2":
        parts = ["lora_te2"]
    else:
        parts = ["lora_te"]

    # Block type
    if nkey.block_type == "mid":
        parts.append("mid_block")
    else:
        parts.append(f"{nkey.block_type}_blocks_{nkey.block_idx}")

    # Layer type
    if nkey.layer_type in ["attn", "attn1", "attn2"]:
        parts.append("attentions")
    elif nkey.layer_type == "ff":
        parts.append("attentions")  # FF is inside attention blocks
    else:
        parts.append("resnets")

    if nkey.layer_idx is not None:
        parts.append(str(nkey.layer_idx))

    # Attention type
    if nkey.layer_type in ["attn1", "attn2", "ff"]:
        parts.append("transformer_blocks_0")
        parts.append(nkey.layer_type)

    # Sublayer
    parts.append(nkey.sublayer.replace(".", "_"))

    # Join with underscores
    base = "_".join(parts)

    # Add LoRA suffix
    lora_suffix = f".lora_{nkey.lora_type}.weight"

    return base + lora_suffix


def _normalized_to_diffusers(nkey) -> str:
    """Convert NormalizedKey to diffusers native format."""
    parts = []

    if nkey.component != "unet":
        parts.append(nkey.component)
    else:
        parts.append("unet")

    # Block type
    if nkey.block_type == "mid":
        parts.append("mid_block")
    else:
        parts.append(f"{nkey.block_type}_blocks")
        parts.append(str(nkey.block_idx))

    # Layer type
    if nkey.layer_type in ["attn", "attn1", "attn2", "ff"]:
        parts.append("attentions")
    else:
        parts.append("resnets")

    if nkey.layer_idx is not None:
        parts.append(str(nkey.layer_idx))

    # Transformer blocks for attention
    if nkey.layer_type in ["attn1", "attn2", "ff"]:
        parts.append("transformer_blocks")
        parts.append("0")
        parts.append(nkey.layer_type)

    # Sublayer
    parts.append(nkey.sublayer)

    # LoRA suffix (A/B format)
    lora_type = "A" if nkey.lora_type == "down" else "B"
    parts.append(f"lora_{lora_type}")
    parts.append("weight")

    return ".".join(parts)