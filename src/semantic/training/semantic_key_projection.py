"""
Semantic key projection for mapping layer keys to semantic groups.

Provides stable, architecture-independent mapping from detailed diffusers layer keys
to semantic importance groups (e.g., self_attention, cross_attention).
"""

from typing import Dict, Optional
import re


def project_layer_key_to_semantic_key(layer_key: str) -> str:
    """
    Project a detailed diffusers layer key to a semantic group.

    Args:
        layer_key: Full layer identifier (e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q")

    Returns:
        Semantic group identifier (e.g., "self_attention.query")

    Examples:
        >>> project_layer_key_to_semantic_key("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q")
        'self_attention.query'
        >>> project_layer_key_to_semantic_key("mid_block.attentions.0.transformer_blocks.0.attn2.to_v")
        'cross_attention.value'
        >>> project_layer_key_to_semantic_key("up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out")
        'self_attention.output'
    """
    # Self-attention (attn1)
    if '.attn1.' in layer_key:
        if '.to_q' in layer_key:
            return 'self_attention.query'
        elif '.to_k' in layer_key:
            return 'self_attention.key'
        elif '.to_v' in layer_key:
            return 'self_attention.value'
        elif '.to_out' in layer_key:
            return 'self_attention.output'
        else:
            return 'self_attention'

    # Cross-attention (attn2)
    elif '.attn2.' in layer_key:
        if '.to_q' in layer_key:
            return 'cross_attention.query'
        elif '.to_k' in layer_key:
            return 'cross_attention.key'
        elif '.to_v' in layer_key:
            return 'cross_attention.value'
        elif '.to_out' in layer_key:
            return 'cross_attention.output'
        else:
            return 'cross_attention'

    # MLP / feedforward
    elif '.ff.' in layer_key or '.mlp.' in layer_key or 'feed_forward' in layer_key:
        return 'mlp'

    # Generic attention (for architectures without attn1/attn2 distinction)
    elif '.attention.' in layer_key or '.attn.' in layer_key:
        if '.to_q' in layer_key:
            return 'attention.query'
        elif '.to_k' in layer_key:
            return 'attention.key'
        elif '.to_v' in layer_key:
            return 'attention.value'
        elif '.to_out' in layer_key:
            return 'attention.output'
        else:
            return 'attention'

    # Fallback: extract block type and depth
    # e.g., "down_blocks" → "down", "up_blocks" → "up", "mid_block" → "mid"
    else:
        if 'down_blocks' in layer_key:
            return 'down'
        elif 'up_blocks' in layer_key:
            return 'up'
        elif 'mid_block' in layer_key:
            return 'mid'
        else:
            return 'other'


def get_semantic_importance(
    semantic_map: Dict,
    layer_key: str,
    feature: str,
    default_importance: float = 0.0,
    debug: bool = False
) -> float:
    """
    Get semantic importance for a layer using projection.

    Args:
        semantic_map: Semantic importance map (feature -> importance_dict)
        layer_key: Full diffusers layer key
        feature: Feature name (e.g., "hair", "face")
        default_importance: Default value if no match found
        debug: Enable debug logging

    Returns:
        Importance score (float)

    Example:
        >>> semantic_map = {"hair": {"self_attention.query": 0.8, "cross_attention": 0.3}}
        >>> get_semantic_importance(semantic_map, "down_blocks.0.attentions.0.attn1.to_q", "hair")
        0.8
    """
    if feature not in semantic_map:
        if debug:
            import logging
            logging.getLogger(__name__).debug(f"Feature '{feature}' not in semantic map")
        return default_importance

    feature_map = semantic_map[feature]

    # Project layer key to semantic group
    semantic_key = project_layer_key_to_semantic_key(layer_key)

    if debug:
        import logging
        logger = logging.getLogger(__name__)
        sample_keys = list(feature_map.keys())[:5]
        logger.info(
            f"Layer '{layer_key}' → semantic key '{semantic_key}'\n"
            f"  Feature map has {len(feature_map)} keys, sample: {sample_keys}"
        )

    # Try exact match
    if semantic_key in feature_map:
        imp = feature_map[semantic_key]
        importance = float(imp) if hasattr(imp, '__float__') else imp
        if debug:
            import logging
            logging.getLogger(__name__).info(f"  ✓ Exact match: {semantic_key} → {importance}")
        return importance

    # Try parent group (e.g., "self_attention.query" → "self_attention")
    if '.' in semantic_key:
        parent_key = semantic_key.split('.')[0]
        if parent_key in feature_map:
            imp = feature_map[parent_key]
            importance = float(imp) if hasattr(imp, '__float__') else imp
            if debug:
                import logging
                logging.getLogger(__name__).info(f"  ✓ Parent match: {parent_key} → {importance}")
            return importance

    # Fallback: uniform importance across all keys in feature
    # (This ensures non-zero signal even with incomplete semantic maps)
    if len(feature_map) > 0:
        # Average importance across all semantic groups
        avg_importance = sum(
            float(v) if hasattr(v, '__float__') else v
            for v in feature_map.values()
        ) / len(feature_map)
        if debug:
            import logging
            logging.getLogger(__name__).info(f"  → Fallback average: {avg_importance}")
        return avg_importance

    if debug:
        import logging
        logging.getLogger(__name__).warning(f"  ✗ No match, using default: {default_importance}")

    return default_importance


# Backward compatibility: alias for old function name
def normalize_semantic_key(layer_key: str) -> str:
    """Alias for project_layer_key_to_semantic_key for backward compatibility."""
    return project_layer_key_to_semantic_key(layer_key)
