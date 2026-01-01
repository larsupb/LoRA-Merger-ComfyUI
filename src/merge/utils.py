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

    Converts high-level filter specification into specific component name sets.
    Architecture-agnostic: works for both Stable Diffusion and DiT LoRAs.

    Args:
        layer_filter: Filter type specification
            - "full": No filtering (returns None)
            - "attn-only": Only attention layers (SD: attn1/attn2, DiT: attention)
            - "mlp-only": Only MLP/feedforward layers (SD: ff, DiT: mlp/feed_forward)
            - "attn-mlp": Both attention and MLP layers

    Returns:
        Set of layer component names to keep, or None for no filtering

    Example:
        >>> parse_layer_filter("attn-mlp")
        {"attn1", "attn2", "attention", "ff", "mlp", "feed_forward"}

    Note:
        This function delegates to LayerFilter.PRESETS for consistency.
        Direct use of the LayerFilter class is recommended for new code.
    """
    # Import here to avoid circular imports
    from ..utils.layer_filter import LayerFilter

    # Delegate to LayerFilter.PRESETS for single source of truth
    return LayerFilter.PRESETS.get(layer_filter, None)


def apply_layer_filter(
    patch_dict: LORA_KEY_DICT,
    layer_filter: Optional[LayerComponentSet],
    detect_architecture: bool = True
) -> LORA_KEY_DICT:
    """
    Apply layer component filter to patch dictionary.

    Filters LoRA patches to include only specified component types.
    Used for selective merging (e.g., merge only attention layers).

    Args:
        patch_dict: Dictionary of layer key -> LoRAAdapter
        layer_filter: Set of component names to keep, or None for no filtering
        detect_architecture: Whether to detect and log architecture (default: True)

    Returns:
        Filtered patch dictionary containing only matching layers

    Note:
        Uses component-based matching (split by '.') to avoid false positives
        from substring matches (e.g., 'ff' in 'diffusion_model').

    Example:
        >>> patches = {"model.attn1.weight": adapter1, "model.ff.weight": adapter2}
        >>> filtered = apply_layer_filter(patches, {"attn1", "attn2"})
        >>> # Returns only {"model.attn1.weight": adapter1}
    """
    from ..utils.layer_filter import detect_lora_architecture

    num_keys = len(patch_dict.keys())

    # Detect architecture before filtering
    if detect_architecture and patch_dict:
        arch_name, arch_meta = detect_lora_architecture(patch_dict)
        total_keys = arch_meta.get("total_keys", num_keys)

        if arch_name != "Unknown":
            logging.info(f"Detected {arch_name} architecture ({total_keys} keys)")
        else:
            logging.debug(f"Processing LoRA ({total_keys} keys, architecture: {arch_name})")

    if layer_filter:
        import re

        def matches_filter(key) -> bool:
            """
            Check if key matches any filter component.

            Uses word-boundary aware matching to avoid false positives.
            For example, 'ff' will match 'ff_net' or '.ff.' but not 'diffusion'.
            """
            # Handle tuple keys from ComfyUI
            key_str = str(key[0]) if isinstance(key, tuple) else str(key)
            key_lower = key_str.lower()

            for filter_pattern in layer_filter:
                pattern_lower = filter_pattern.lower()

                # Create regex pattern with word boundaries
                # Match pattern when surrounded by dots, underscores, or at start/end
                regex_pattern = r'(?:^|[._])' + re.escape(pattern_lower) + r'(?:[._]|$)'

                if re.search(regex_pattern, key_lower):
                    return True

            return False

        patch_dict = {
            k0: v0
            for k0, v0 in patch_dict.items()
            if matches_filter(k0)
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
    nearswap). Extracted to eliminate code duplication.

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


def simple_weighted_average(
    tensors: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    normalize: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Compute simple weighted average of tensors.

    Used for CLIP layer merging, provides straightforward linear interpolation
    between multiple LoRA tensors.

    Args:
        tensors: Dictionary mapping LoRA names to their tensors
        weights: Dictionary mapping LoRA names to their weight values
        normalize: Whether to normalize weights to sum to 1 (default: True)
                  If False, uses raw weighted sum
        device: Optional device to perform computation on
        dtype: Optional dtype for computation

    Returns:
        Weighted average tensor

    Example:
        >>> tensors = {
        ...     "lora1": torch.tensor([1.0, 2.0, 3.0]),
        ...     "lora2": torch.tensor([4.0, 5.0, 6.0])
        ... }
        >>> weights = {"lora1": 0.3, "lora2": 0.7}
        >>> result = simple_weighted_average(tensors, weights)
        >>> # result ≈ [3.1, 4.1, 5.1] (0.3 * [1,2,3] + 0.7 * [4,5,6])

    Note:
        All tensors must have the same shape. If normalize=True and weights sum
        to zero, returns zeros with the same shape as input tensors.
    """
    if not tensors:
        raise ValueError("Cannot compute weighted average of empty tensor dict")

    # Ensure all LoRAs in tensors have weights
    missing_weights = set(tensors.keys()) - set(weights.keys())
    if missing_weights:
        raise ValueError(f"Missing weights for LoRAs: {missing_weights}")

    # Move tensors to device/dtype if specified
    if device is not None or dtype is not None:
        tensors = {
            name: tensor.to(device=device if device else tensor.device,
                           dtype=dtype if dtype else tensor.dtype)
            for name, tensor in tensors.items()
        }

    # Extract weights in same order as tensors
    weight_list = [weights[name] for name in tensors.keys()]
    weight_sum = sum(weight_list)

    # Compute weighted sum
    result = None
    for name, weight_val in zip(tensors.keys(), weight_list):
        weighted_tensor = tensors[name] * weight_val
        if result is None:
            result = weighted_tensor
        else:
            result = result + weighted_tensor

    # Normalize if requested
    if normalize and weight_sum != 0:
        result = result / weight_sum
    elif normalize and weight_sum == 0:
        # If all weights are zero, return zeros
        logging.warning("All weights sum to zero, returning zero tensor")
        result = torch.zeros_like(next(iter(tensors.values())))

    return result
