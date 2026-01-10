"""
Feature isolation prompts for semantic LoRA analysis.

This module defines prompts designed to isolate specific character features
during gradient-based attribution analysis.
"""

from typing import Dict, List

# Feature categories and their isolation prompts
FEATURE_PROMPTS: Dict[str, List[str]] = {
    "hair": [
        "1girl, focus on hair, simple background, detailed hair",
        "close-up portrait, hair visible, hair strands detailed",
        "character with visible hairstyle, hair color clear",
    ],
    "eyes": [
        "1girl, focus on eyes, simple background, detailed eyes",
        "close-up portrait, eyes visible, eye details clear",
        "character with visible eyes, eye color clear",
    ],
    "face": [
        "1girl, focus on face, simple background, facial features detailed",
        "portrait, face visible, facial structure clear",
        "character face, facial characteristics detailed",
    ],
    "clothing": [
        "1girl, focus on outfit, simple background, clothing detailed",
        "full body, clothing visible, outfit details clear",
        "character wearing outfit, clothing style visible",
    ],
    "accessories": [
        "1girl, focus on accessories, simple background, accessories detailed",
        "character with accessories, jewelry visible, ornaments clear",
        "accessories and decorations visible, detailed ornaments",
    ],
    "body": [
        "1girl, focus on body, simple background, body proportions visible",
        "full body, body shape clear, proportions visible",
        "character body, physical characteristics clear",
    ],
}

# Architecture-specific prompt adjustments
ARCHITECTURE_ADJUSTMENTS: Dict[str, Dict[str, str]] = {
    "sd15": {
        "prefix": "masterpiece, best quality, ",
        "suffix": ", high quality, detailed",
    },
    "sdxl": {
        "prefix": "",
        "suffix": ", detailed, high resolution",
    },
    "dit": {
        "prefix": "",
        "suffix": ", clear, detailed",
    },
}

# Feature groups for coarse-grained analysis
FEATURE_GROUPS: Dict[str, List[str]] = {
    "head": ["hair", "eyes", "face"],
    "body": ["body", "clothing"],
    "details": ["accessories"],
}


def get_feature_prompts(
    feature: str,
    architecture: str = "sdxl",
    num_prompts: int = 3,
) -> List[str]:
    """
    Get feature isolation prompts for a specific feature.

    Supports both predefined features (hair, eyes, etc.) and custom feature names.
    For custom features, generates generic prompts based on the feature name.

    Args:
        feature: Feature name (hair, eyes, clothing, uniform, background, etc.)
        architecture: Model architecture (sd15, sdxl, dit)
        num_prompts: Number of prompts to return (default: 3)

    Returns:
        List of formatted prompts for the feature

    Examples:
        >>> get_feature_prompts("hair")  # Predefined
        ["1girl, focus on hair, simple background, detailed hair", ...]

        >>> get_feature_prompts("uniform")  # Custom
        ["1girl, focus on uniform, simple background, detailed uniform", ...]
    """
    # Check if predefined feature
    if feature in FEATURE_PROMPTS:
        base_prompts = FEATURE_PROMPTS[feature][:num_prompts]
    else:
        # Generate generic prompts for custom feature
        base_prompts = _generate_custom_feature_prompts(feature, num_prompts)

    # Apply architecture-specific adjustments
    if architecture in ARCHITECTURE_ADJUSTMENTS:
        adj = ARCHITECTURE_ADJUSTMENTS[architecture]
        prefix = adj["prefix"]
        suffix = adj["suffix"]
        return [f"{prefix}{prompt}{suffix}" for prompt in base_prompts]

    return base_prompts


def _generate_custom_feature_prompts(feature: str, num_prompts: int = 3) -> List[str]:
    """
    Generate generic prompts for a custom feature name.

    Args:
        feature: Custom feature name (e.g., "uniform", "background", "weapon")
        num_prompts: Number of prompts to generate

    Returns:
        List of generic prompts featuring the custom feature
    """
    # Template variations for custom features
    templates = [
        f"1girl, focus on {feature}, simple background, detailed {feature}",
        f"close-up, {feature} visible, {feature} details clear",
        f"character with visible {feature}, {feature} clearly shown",
        f"portrait, {feature} in view, detailed {feature}",
        f"full body, {feature} visible, {feature} details",
    ]

    # Return requested number of prompts (cycle if needed)
    result = []
    for i in range(num_prompts):
        result.append(templates[i % len(templates)])

    return result


def get_all_features() -> List[str]:
    """Get list of all available features."""
    return list(FEATURE_PROMPTS.keys())


def get_feature_group(group_name: str) -> List[str]:
    """
    Get features belonging to a feature group.

    Args:
        group_name: Name of feature group (head, body, details)

    Returns:
        List of feature names in the group
    """
    if group_name not in FEATURE_GROUPS:
        available = ", ".join(FEATURE_GROUPS.keys())
        raise ValueError(
            f"Unknown feature group '{group_name}'. Available groups: {available}"
        )

    return FEATURE_GROUPS[group_name]
