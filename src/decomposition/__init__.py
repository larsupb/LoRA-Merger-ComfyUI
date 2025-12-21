"""
Decomposition module for LoRA Power-Merger.

Handles LoRA decomposition operations including:
- Decomposition of LoRAs into (up, down, alpha) tuples
- Caching layer for expensive decomposition operations
- Base classes for decomposition strategies (SVD, QR, etc.)
"""

from .base import (
    TensorDecomposer,
    DecompositionMethod,
    SingularValueDistribution,
)
from .svd import (
    SVDDecomposer,
    RandomizedSVDDecomposer,
    EnergyBasedRandomizedSVDDecomposer,
    QRDecomposer,
)

__all__ = [
    # Base classes
    'TensorDecomposer',
    'DecompositionMethod',
    'SingularValueDistribution',
    # Decomposers
    'SVDDecomposer',
    'RandomizedSVDDecomposer',
    'EnergyBasedRandomizedSVDDecomposer',
    'QRDecomposer',
]
