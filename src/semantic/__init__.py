"""
Semantic LoRA analysis module.

This module provides tools for analyzing LoRAs to understand which weights
contribute to which semantic features (hair, eyes, clothing, etc.).

Key components:
- Gradient-based semantic analysis
- Feature importance scoring
- Semantic map caching

Usage:
    from src.analysis import (
        GradientSemanticAnalyzer,
        get_feature_prompts,
        get_cache,
    )

    # Analyze a LoRA
    analyzer = GradientSemanticAnalyzer(clip, device, dtype)
    semantic_map = analyzer.analyze_lora(...)

    # Cache results
    cache = get_cache()
    cache.set(lora_path, config, semantic_map, metadata)
"""

from .cache import SemanticMapCache, get_cache
from .feature_prompts import (
    FEATURE_PROMPTS,
    FEATURE_GROUPS,
    get_feature_prompts,
    get_all_features,
    get_feature_group,
)
from .gradient_analyzer import GradientSemanticAnalyzer

__all__ = [
    # Gradient analysis
    "GradientSemanticAnalyzer",
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
