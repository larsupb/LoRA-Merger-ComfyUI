"""
Semantic LoRA analysis module.

This module provides tools for analyzing LoRAs to understand which weights
contribute to which semantic features (hair, eyes, clothing, etc.).

Key components:
- Gradient-based attribution (Integrated Gradients)
- Cross-attention flow tracking
- Feature importance scoring
- Semantic map caching

Usage:
    from src.analysis import (
        IntegratedGradientsAttributor,
        AttentionFlowTracker,
        get_feature_prompts,
        get_cache,
    )

    # Analyze a LoRA
    attributor = IntegratedGradientsAttributor(model, device)
    attributions = attributor.attribute_lora_weights(...)

    # Cache results
    cache = get_cache()
    cache.set(lora_path, lora_patches, config, semantic_map)
"""

from .attention import (
    AttentionFlowTracker,
    FeatureAttentionAnalyzer,
    map_attention_to_lora_layers,
    compute_spatial_feature_attribution,
)
from .cache import SemanticMapCache, get_cache
from .feature_prompts import (
    FEATURE_PROMPTS,
    FEATURE_GROUPS,
    get_feature_prompts,
    get_all_features,
    get_feature_group,
)
from .gradient_analyzer import GradientSemanticAnalyzer
from .gradients import (
    IntegratedGradientsAttributor,
    SimpleGradientAttributor,
    compute_weight_importance,
    normalize_importance_scores,
)

__all__ = [
    # Attribution
    "IntegratedGradientsAttributor",
    "SimpleGradientAttributor",
    "compute_weight_importance",
    "normalize_importance_scores",
    "GradientSemanticAnalyzer",
    # Attention tracking
    "AttentionFlowTracker",
    "FeatureAttentionAnalyzer",
    "map_attention_to_lora_layers",
    "compute_spatial_feature_attribution",
    # Caching
    "SemanticMapCache",
    "get_cache",
    # Feature prompts
    "FEATURE_PROMPTS",
    "FEATURE_GROUPS",
    "get_feature_prompts",
    "get_all_features",
    "get_feature_group",
]
