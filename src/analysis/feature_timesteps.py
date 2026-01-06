"""
Feature-specific timestep configuration for gradient analysis.

Different semantic features are best captured at different noise levels:
- Early timesteps (high noise): composition, layout
- Middle timesteps: main features (face, hair, clothing)
- Late timesteps (low noise): fine details, textures

"""

from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TimestepConfig:
    """Configuration for multi-timestep analysis."""
    timesteps: List[int]
    weights: List[float]

    def __post_init__(self):
        assert len(self.timesteps) == len(self.weights), \
            "timesteps and weights must have same length"
        # Normalize weights to sum to 1
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]


# Default multi-timestep configuration

# Weighted towards middle timesteps where features are most distinct

DEFAULT_MULTI_TIMESTEP = TimestepConfig(
    timesteps=[750, 500, 250],
    weights=[0.25, 0.50, 0.25],  # Middle weighted 2x
)

# Single timestep (original behavior)

SINGLE_TIMESTEP = TimestepConfig(
    timesteps=[500],
    weights=[1.0],
)

# Feature-specific optimal timesteps

# Based on when different semantic features are "decided" during denoising

FEATURE_TIMESTEP_CONFIGS: Dict[str, TimestepConfig] = {
    # Composition/Layout features - captured early
    "composition": TimestepConfig([850, 700, 550], [0.4, 0.4, 0.2]),
    "pose": TimestepConfig([800, 650, 500], [0.35, 0.40, 0.25]),
    "background": TimestepConfig([800, 600, 400], [0.40, 0.35, 0.25]),

    # Main features - captured in middle
    "face": TimestepConfig([650, 500, 350], [0.25, 0.50, 0.25]),
    "hair": TimestepConfig([600, 450, 300], [0.25, 0.50, 0.25]),
    "clothing": TimestepConfig([650, 500, 350], [0.25, 0.50, 0.25]),
    "body": TimestepConfig([700, 550, 400], [0.30, 0.45, 0.25]),

    # Detail features - captured later
    "eyes": TimestepConfig([500, 350, 200], [0.25, 0.45, 0.30]),
    "texture": TimestepConfig([450, 300, 150], [0.20, 0.40, 0.40]),
    "accessories": TimestepConfig([550, 400, 250], [0.25, 0.45, 0.30]),
    "hands": TimestepConfig([500, 350, 200], [0.25, 0.45, 0.30]),

    # Style features - broad range
    "style": TimestepConfig([750, 500, 250], [0.33, 0.34, 0.33]),
    "lighting": TimestepConfig([700, 500, 300], [0.30, 0.40, 0.30]),
    "color": TimestepConfig([600, 450, 300], [0.30, 0.40, 0.30]),
}


def get_timestep_config(
        feature: str,
        use_feature_specific: bool = True,
        use_multi_timestep: bool = True,
) -> TimestepConfig:
    """
    Get timestep configuration for a feature.

    Args:
        feature: Feature name (e.g., "hair", "clothing")
        use_feature_specific: If True, use feature-optimized timesteps
        use_multi_timestep: If True, use multiple timesteps; if False, single t=500

    Returns:
        TimestepConfig with timesteps and weights
    """
    if not use_multi_timestep:
        return SINGLE_TIMESTEP

    if use_feature_specific and feature.lower() in FEATURE_TIMESTEP_CONFIGS:
        return FEATURE_TIMESTEP_CONFIGS[feature.lower()]

    return DEFAULT_MULTI_TIMESTEP