"""
Semantic adapter modules for learning optimal LoRA composition.

This package provides trainable adapters that learn to combine multiple
LoRAs based on semantic importance maps, enabling feature-aware composition
beyond simple linear interpolation.
"""

from .base import BaseAdapter
from .rank_space import RankSpaceAdapter
from .output_space import OutputSpaceAdapter
from .registry import AdapterRegistry
from .hook_manager import ForwardHookManager

__all__ = [
    "BaseAdapter",
    "RankSpaceAdapter",
    "OutputSpaceAdapter",
    "AdapterRegistry",
    "ForwardHookManager",
]
