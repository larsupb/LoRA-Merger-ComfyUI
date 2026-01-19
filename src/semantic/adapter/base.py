"""
Base adapter interface for semantic LoRA composition.

Adapters learn to combine multiple LoRA deltas based on semantic importance maps.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torch.nn as nn


class BaseAdapter(nn.Module, ABC):
    """
    Abstract base class for semantic adapters.

    Adapters learn to combine multiple LoRA deltas based on
    semantic importance maps. This enables learned, feature-aware
    composition that goes beyond simple linear interpolation.

    Two main adapter types:
    - RankSpaceAdapter: Operates in low-rank subspace (attention/MLP layers)
    - OutputSpaceAdapter: Operates on full deltas (conv/unknown layers)
    """

    def __init__(
        self,
        layer_key: str,
        semantic_dim: int,
        feature_names: List[str],
    ):
        """
        Initialize base adapter.

        Args:
            layer_key: Layer identifier (e.g., "diffusion_model.layers.0.attention.q")
            semantic_dim: Dimensionality of semantic feature vector
            feature_names: List of feature names (e.g., ["hair", "eyes", "clothing"])
        """
        super().__init__()
        self.layer_key = layer_key
        self.semantic_dim = semantic_dim
        self.feature_names = feature_names

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Combine multiple LoRA outputs using semantic guidance.

        Returns:
            Combined delta tensor
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, float]:
        """
        Return adapter statistics for monitoring.

        Returns:
            Dict of statistics (e.g., gate sparsity, residual norms)
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layer_key={self.layer_key}, "
            f"semantic_dim={self.semantic_dim}, "
            f"num_features={len(self.feature_names)})"
        )
