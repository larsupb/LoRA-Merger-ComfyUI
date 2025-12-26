"""
Merge module for LoRA Power-Merger.

This module contains all merge-related functionality:
- Core merger logic (LoraMergerMergekit node)
- Merge algorithm implementations
- Helper utilities for merging operations
- Algorithm dispatcher
- Base classes for merge method nodes
- Custom sparsification methods
"""

from .utils import (
    create_map,
    create_tensor_param,
    parse_layer_filter,
    apply_layer_filter,
    apply_weights_to_tensors,
    simple_weighted_average,
)
from .dispatcher import get_merge_method, prepare_method_args
from .algorithms import MERGE_ALGORITHMS, get_merge_algorithm
from .base_node import BaseMergeMethodNode, BaseTaskArithmeticNode

__all__ = [
    # Utils
    'create_map',
    'create_tensor_param',
    'parse_layer_filter',
    'apply_layer_filter',
    'apply_weights_to_tensors',
    'simple_weighted_average',
    # Dispatcher
    'get_merge_method',
    'prepare_method_args',
    # Algorithms
    'MERGE_ALGORITHMS',
    'get_merge_algorithm',
    # Base classes
    'BaseMergeMethodNode',
    'BaseTaskArithmeticNode',
]
