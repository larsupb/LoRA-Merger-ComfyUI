import re
from typing import Dict, Any, Optional


def detect_block_names(layer_key, layers_per_group: int = 5) -> Optional[Dict[str, Any]]:
    """
    Detect block names for DiT (Diffusion Transformer) architecture.

    DiT models have a flat structure with sequential transformer layers (e.g., layers.0 through layers.39).
    This function groups them into "main blocks" for easier block-wise manipulation.

    Args:
        layer_key: The layer key (string or tuple). If tuple, first element is used as the key string.
                  (e.g., "diffusion_model.layers.13.attention.qkv.weight" or
                   ("diffusion_model.layers.13.attention.qkv.weight", 0))
        layers_per_group: Number of layers to group into one main block (default: 5)
                         For a 40-layer model, this creates 8 main blocks

    Returns:
        Dictionary with block information, or None if the key doesn't match DiT pattern

    Examples:
        >>> detect_block_names("diffusion_model.layers.13.attention.qkv.weight", layers_per_group=5)
        {
            "layer_idx": "13",
            "component": "attention",
            "main_block": "layers_group.2",  # layers 10-14 = group 2
            "sub_block": "layers.13"
        }
    """
    # Convert tuple keys to strings (ComfyUI uses tuple keys)
    if isinstance(layer_key, tuple):
        layer_key = layer_key[0]

    # DiT pattern: diffusion_model.layers.{idx}.{component}
    # Components typically include: attention, norm1, norm2, mlp, feed_forward, etc.
    exp_dit = re.compile(r"""
        (?:diffusion_model\.)?                     # optional prefix
        layers\.
        (?P<layer_idx>\d+)                         # layer index (0-39 for 40-layer model)
        \.
        (?P<component>[a-zA-Z_][a-zA-Z0-9_]*)      # component type (any valid identifier)
        (?:\..+)?                                  # allow nested submodules (e.g. .weight, .to_q.weight)
    """, re.VERBOSE)

    match = exp_dit.search(layer_key)
    if match:
        layer_idx = int(match.group("layer_idx"))
        component = match.group("component")

        # Calculate which group this layer belongs to
        # For layers_per_group=5: layers 0-4 -> group 0, layers 5-9 -> group 1, etc.
        group_idx = layer_idx // layers_per_group

        out = {
            "layer_idx": match.group("layer_idx"),
            "component": component,
            "main_block": f"layers_group.{group_idx}",
            "sub_block": f"layers.{layer_idx}",
            "group_idx": group_idx,
            "group_start": group_idx * layers_per_group,
            "group_end": (group_idx + 1) * layers_per_group - 1,
        }
        return out

    return None


def get_group_count(total_layers: int, layers_per_group: int = 5) -> int:
    """
    Calculate the number of groups for a given number of layers.

    Args:
        total_layers: Total number of transformer layers in the model
        layers_per_group: Number of layers per group

    Returns:
        Number of groups

    Examples:
        >>> get_group_count(40, 5)
        8
        >>> get_group_count(28, 5)
        6
    """
    return (total_layers + layers_per_group - 1) // layers_per_group


def detect_architecture(patch_dict: Dict) -> Optional[str]:
    """
    Auto-detect if a LoRA uses DiT architecture by examining its keys.

    Args:
        patch_dict: Dictionary of LoRA patches

    Returns:
        "dit" if DiT architecture is detected, None otherwise
    """
    # Sample a few keys to check
    sample_size = min(10, len(patch_dict))
    sample_keys = list(patch_dict.keys())[:sample_size]

    dit_pattern = re.compile(r"(?:diffusion_model\.)?layers\.\d+\.")

    dit_matches = sum(1 for key in sample_keys if dit_pattern.search(str(key)))

    # If more than 50% of sampled keys match DiT pattern, consider it DiT
    if dit_matches / sample_size > 0.5:
        return "dit"

    return None
