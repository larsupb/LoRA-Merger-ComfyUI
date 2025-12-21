"""
Utility functions for merge operations.

Contains helper functions used throughout the merging pipeline:
- WeightInfo mapping creation
- Tensor parameter creation
- Layer filtering logic
"""

import logging
from typing import Dict, Optional, Any

import torch
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ImmutableMap

from ..types import (
    LORA_KEY_DICT,
    LayerFilterType,
    LayerComponentSet,
)


def create_map(
    key: str,
    tensors: Dict[ModelReference, torch.Tensor],
    dtype: torch.dtype
) -> ImmutableMap[ModelReference, WeightInfo]:
    """
    Create an ImmutableMap of WeightInfo objects for mergekit operations.

    Args:
        key: Layer key identifier
        tensors: Dictionary mapping model references to tensors
        dtype: Data type for the weight info

    Returns:
        ImmutableMap of WeightInfo objects for each model reference
    """
    return ImmutableMap({
        r: WeightInfo(name=f'model{i}.{key}', dtype=dtype)
        for i, r in enumerate(tensors.keys())
    })


def create_tensor_param(tensor_weight: float, method_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create tensor parameter dictionary for merge operations.

    Combines weight value with method-specific arguments into a single
    parameter dictionary.

    Args:
        tensor_weight: Weight/strength value for the tensor
        method_args: Additional method-specific arguments (density, epsilon, etc.)

    Returns:
        Dictionary with weight and method args merged

    Example:
        >>> create_tensor_param(0.5, {"density": 0.8})
        {"weight": 0.5, "density": 0.8}
    """
    out = {"weight": tensor_weight}
    out.update(method_args)
    return out


def parse_layer_filter(layer_filter: LayerFilterType) -> Optional[LayerComponentSet]:
    """
    Parse layer filter string into set of component names.

    Converts high-level filter specification into specific component name sets
    for Stable Diffusion architecture LoRAs.

    Args:
        layer_filter: Filter type specification
            - "full": No filtering (returns None)
            - "attn-mlp": Only attention and MLP layers
            - "attn-only": Only attention layers

    Returns:
        Set of layer component names to keep, or None for no filtering

    Example:
        >>> parse_layer_filter("attn-mlp")
        {"attn1", "attn2", "ff"}
    """
    if layer_filter == "full":
        return None
    elif layer_filter == "attn-mlp":
        return {"attn1", "attn2", "ff"}
    elif layer_filter == "attn-only":
        return {"attn1", "attn2"}
    return None


def apply_layer_filter(
    patch_dict: LORA_KEY_DICT,
    layer_filter: Optional[LayerComponentSet]
) -> LORA_KEY_DICT:
    """
    Apply layer component filter to patch dictionary.

    Filters LoRA patches to include only specified component types.
    Used for selective merging (e.g., merge only attention layers).

    Args:
        patch_dict: Dictionary of layer key -> LoRAAdapter
        layer_filter: Set of component names to keep, or None for no filtering

    Returns:
        Filtered patch dictionary containing only matching layers

    Example:
        >>> patches = {"attn1.weight": adapter1, "ff.weight": adapter2, "other": adapter3}
        >>> filtered = apply_layer_filter(patches, {"attn1", "attn2"})
        >>> # Returns only {"attn1.weight": adapter1}
    """
    num_keys = len(patch_dict.keys())

    if layer_filter:
        patch_dict = {
            k0: v0
            for k0, v0 in patch_dict.items()
            if any(layer in k0 for layer in layer_filter)
        }

    logging.info(
        f"Stacking {len(patch_dict)} keys with {num_keys - len(patch_dict)} "
        f"filtered out by filter method {layer_filter}."
    )

    return patch_dict


def apply_weights_to_tensors(
    tensors: Dict[str, torch.Tensor],
    tensor_parameters: Dict[str, Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """
    Apply strength weights to tensors.

    This is a common pattern across multiple merge algorithms (slerp, karcher,
    nearswap, arcee_fusion). Extracted to eliminate code duplication.

    Args:
        tensors: Dictionary mapping LoRA names to their tensors
        tensor_parameters: Dictionary mapping LoRA names to their parameters
                          (must contain "weight" key)

    Returns:
        Dictionary mapping LoRA names to weighted tensors

    Example:
        >>> tensors = {"lora1": torch.ones(10, 10), "lora2": torch.ones(10, 10)}
        >>> params = {"lora1": {"weight": 0.5}, "lora2": {"weight": 0.8}}
        >>> weighted = apply_weights_to_tensors(tensors, params)
        >>> # lora1 scaled by 0.5, lora2 scaled by 0.8
    """
    return {
        ref: tensor_parameters[ref]["weight"] * tensors[ref]
        for ref in tensors.keys()
    }
