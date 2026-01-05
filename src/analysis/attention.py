"""
Cross-attention flow analysis for semantic feature attribution.

Tracks attention patterns to identify which LoRA weights influence
which text tokens and image regions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionFlowTracker:
    """
    Tracks cross-attention patterns during diffusion model forward pass.

    Useful for identifying which LoRA weights influence attention to
    specific text tokens (e.g., "hair", "clothing").
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize attention tracker.

        Args:
            model: Diffusion model to track
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def register_attention_hooks(self, layer_pattern: str = "attn") -> None:
        """
        Register forward hooks to capture attention maps.

        Args:
            layer_pattern: Pattern to match attention layers (default: "attn")
        """
        self.hooks.clear()

        def hook_fn(name):
            def hook(module, input, output):
                # Store attention map
                # Assumes output contains attention weights
                # Format varies by architecture - this is a simplified version
                if isinstance(output, tuple):
                    self.attention_maps[name] = output[1]  # Attention weights
                else:
                    self.attention_maps[name] = output
            return hook

        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if layer_pattern in name.lower():
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)

        logger.info(f"Registered {len(self.hooks)} attention hooks")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """
        Get captured attention maps.

        Returns:
            Dictionary mapping layer names to attention tensors
        """
        return self.attention_maps.copy()

    def compute_token_attribution(
        self,
        attention_maps: Dict[str, torch.Tensor],
        token_index: int,
        aggregation: str = "mean",
    ) -> Dict[str, float]:
        """
        Compute how much each layer attends to a specific token.

        Args:
            attention_maps: Attention maps from forward pass
            token_index: Index of target token in text embedding
            aggregation: How to aggregate attention (mean, max, sum)

        Returns:
            Attribution score per layer
        """
        attributions = {}

        for layer_name, attn_map in attention_maps.items():
            # attn_map shape: [batch, heads, spatial, text_tokens]
            # We want attention to specific token across all spatial positions

            if attn_map.dim() == 4:
                # [batch, heads, spatial, tokens]
                token_attention = attn_map[:, :, :, token_index]

                if aggregation == "mean":
                    score = token_attention.mean().item()
                elif aggregation == "max":
                    score = token_attention.max().item()
                elif aggregation == "sum":
                    score = token_attention.sum().item()
                else:
                    score = token_attention.mean().item()

                attributions[layer_name] = score

        return attributions

    def compute_attention_rollout(
        self,
        attention_maps: Dict[str, torch.Tensor],
        token_index: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attention rollout for a specific token.

        Attention rollout propagates attention through multiple layers
        to get cumulative attribution.

        Args:
            attention_maps: Attention maps from forward pass
            token_index: Index of target token

        Returns:
            Rolled-out attention maps per layer

        Reference:
            Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)
        """
        # Sort layers by depth
        sorted_layers = sorted(attention_maps.keys())

        rollout_maps = {}
        cumulative_attention = None

        for layer_name in sorted_layers:
            attn_map = attention_maps[layer_name]

            # Extract attention to target token
            # Shape: [batch, heads, spatial, tokens] -> [batch, heads, spatial]
            if attn_map.dim() == 4:
                token_attn = attn_map[:, :, :, token_index]

                # Average over heads: [batch, spatial]
                token_attn = token_attn.mean(dim=1)

                if cumulative_attention is None:
                    cumulative_attention = token_attn
                else:
                    # Multiply with previous layers (attention composition)
                    cumulative_attention = cumulative_attention * token_attn

                rollout_maps[layer_name] = cumulative_attention.clone()

        return rollout_maps


class FeatureAttentionAnalyzer:
    """
    Analyzes which LoRA weights are important for specific semantic features
    based on attention patterns.
    """

    def __init__(self, tracker: AttentionFlowTracker):
        """
        Initialize analyzer.

        Args:
            tracker: Attention flow tracker
        """
        self.tracker = tracker

    def analyze_feature_importance(
        self,
        feature_token_indices: Dict[str, List[int]],
        lora_patches: Dict[str, Any],
        num_samples: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which LoRA layers are important for each feature.

        Args:
            feature_token_indices: Mapping from feature names to token indices
                Example: {"hair": [3, 4], "clothing": [8, 9]}
            lora_patches: LoRA patch dictionary
            num_samples: Number of forward passes to average over

        Returns:
            Importance scores: {feature: {layer_key: importance}}

        Note:
            This is a simplified placeholder. Full implementation requires
            running actual forward passes with different prompts.
        """
        feature_importance = {}

        # This is a placeholder - actual implementation would:
        # 1. Run forward passes with feature-specific prompts
        # 2. Capture attention maps
        # 3. Compute token attribution for each feature
        # 4. Map attention layers back to LoRA weight layers
        # 5. Aggregate across multiple samples

        for feature, token_indices in feature_token_indices.items():
            feature_importance[feature] = {}

            # Placeholder: assign uniform importance
            for layer_key in lora_patches.keys():
                feature_importance[feature][layer_key] = 1.0 / len(lora_patches)

        logger.warning(
            "FeatureAttentionAnalyzer.analyze_feature_importance is a placeholder"
        )

        return feature_importance


def map_attention_to_lora_layers(
    attention_layer_names: List[str],
    lora_layer_keys: List[str],
) -> Dict[str, str]:
    """
    Map attention layer names to LoRA layer keys.

    Args:
        attention_layer_names: Names of attention layers (from hooks)
        lora_layer_keys: Keys in LoRA patches dictionary

    Returns:
        Mapping from attention layer to LoRA layer

    Example:
        >>> attn_layers = ["model.layers.5.attention"]
        >>> lora_keys = ["diffusion_model.layers.5.attention.qkv"]
        >>> mapping = map_attention_to_lora_layers(attn_layers, lora_keys)
        >>> # {"model.layers.5.attention": "diffusion_model.layers.5.attention.qkv"}
    """
    mapping = {}

    for attn_layer in attention_layer_names:
        # Simple heuristic: find LoRA key with most overlapping path components
        best_match = None
        best_score = 0

        attn_parts = attn_layer.split(".")

        for lora_key in lora_layer_keys:
            lora_parts = lora_key.split(".")

            # Count matching parts
            score = sum(1 for a, l in zip(attn_parts, lora_parts) if a == l)

            if score > best_score:
                best_score = score
                best_match = lora_key

        if best_match is not None:
            mapping[attn_layer] = best_match

    return mapping


def compute_spatial_feature_attribution(
    attention_map: torch.Tensor,
    feature_region: Tuple[int, int, int, int],
) -> float:
    """
    Compute attention attribution for a spatial region.

    Useful for features localized to specific image regions
    (e.g., "tattoo on arm" -> specific spatial location).

    Args:
        attention_map: Attention map [batch, heads, spatial, tokens]
        feature_region: (x1, y1, x2, y2) bounding box in spatial grid

    Returns:
        Average attention in the region
    """
    # This is a placeholder - actual implementation would:
    # 1. Convert spatial indices to grid coordinates
    # 2. Extract attention for that region
    # 3. Average across the region

    # For now, return mean of all attention
    return attention_map.mean().item()
