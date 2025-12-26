"""
Utilities package for LoRA Power-Merger.

Provides helper utilities including:
- Layer filtering
- Architecture detection
- Progress tracking
- Configuration constants
- Spectral norm regularization
"""

from .layer_filter import LayerFilter, detect_lora_architecture, is_clip_layer
from .progress import ThreadSafeProgressBar
from .spectral_norm import (
    spectral_norm,
    apply_spectral_norm,
    compute_spectral_norms,
)
from .config import *

__all__ = [
    'LayerFilter',
    'detect_lora_architecture',
    'is_clip_layer',
    'ThreadSafeProgressBar',
    'spectral_norm',
    'apply_spectral_norm',
    'compute_spectral_norms',
]
