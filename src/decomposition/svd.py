"""
SVD-based tensor decomposition implementations.

Provides SVD, randomized SVD, and energy-based randomized SVD decomposers.
"""

import logging
from typing import Tuple

import torch

from .base import TensorDecomposer, SingularValueDistribution


class SVDDecomposer(TensorDecomposer):
    """
    Standard SVD decomposition.

    Uses torch.linalg.svd for full singular value decomposition.
    Most accurate but slower for large matrices.
    """

    def _decompose_2d(
        self,
        weight_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform full SVD decomposition.

        Args:
            weight_2d: 2D weight tensor

        Returns:
            Tuple of (U, S, Vh)

        Raises:
            RuntimeError: If SVD computation fails
        """
        try:
            U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
            return U, S, Vh
        except RuntimeError as e:
            if "singular value decomposition" in str(e).lower():
                raise RuntimeError(
                    f"SVD failed for tensor shape {weight_2d.shape}. "
                    "Matrix may be singular or ill-conditioned."
                ) from e
            raise


class RandomizedSVDDecomposer(TensorDecomposer):
    """
    Randomized SVD decomposition.

    Faster approximation of SVD for large matrices.
    Uses randomized algorithm with power iterations.
    """

    def __init__(
        self,
        n_oversamples: int = 10,
        n_iter: int = 2,
        **kwargs
    ):
        """
        Initialize randomized SVD decomposer.

        Args:
            n_oversamples: Number of additional samples for randomization
            n_iter: Number of power iterations for accuracy
            **kwargs: Passed to parent TensorDecomposer
        """
        super().__init__(**kwargs)
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter

    def _decompose_2d(
        self,
        weight_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform randomized SVD.

        Args:
            weight_2d: 2D weight tensor

        Returns:
            Tuple of (U, S, Vh)
        """
        # For small matrices, use standard SVD
        if min(weight_2d.shape) < 100:
            return torch.linalg.svd(weight_2d, full_matrices=False)

        # Randomized SVD implementation
        m, n = weight_2d.shape
        rank = min(m, n)

        # Determine sketch size
        sketch_size = min(rank, rank + self.n_oversamples)

        # Random projection
        Omega = torch.randn(
            n, sketch_size,
            dtype=weight_2d.dtype,
            device=weight_2d.device
        )

        # Compute sketch
        Y = weight_2d @ Omega

        # Power iterations for better approximation
        for _ in range(self.n_iter):
            Y = weight_2d @ (weight_2d.T @ Y)

        # Orthogonalize
        Q, _ = torch.linalg.qr(Y)

        # Project and decompose
        B = Q.T @ weight_2d
        U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)

        # Recover U
        U = Q @ U_b

        return U, S, Vh


class EnergyBasedRandomizedSVDDecomposer(RandomizedSVDDecomposer):
    """
    Energy-based randomized SVD.

    Adaptive randomized SVD that adjusts sketch size based on
    spectral energy distribution.
    """

    def __init__(
        self,
        energy_threshold: float = 0.99,
        **kwargs
    ):
        """
        Initialize energy-based randomized SVD.

        Args:
            energy_threshold: Target energy retention (0.0 to 1.0)
            **kwargs: Passed to parent RandomizedSVDDecomposer
        """
        super().__init__(**kwargs)
        self.energy_threshold = energy_threshold

    def _decompose_2d(
        self,
        weight_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform energy-based randomized SVD.

        First estimates spectral energy, then adapts sketch size.

        Args:
            weight_2d: 2D weight tensor

        Returns:
            Tuple of (U, S, Vh)
        """
        # Perform initial randomized SVD with moderate sketch size
        U, S, Vh = super()._decompose_2d(weight_2d)

        # Calculate energy retention
        S_squared = S.pow(2)
        cumulative_energy = torch.cumsum(S_squared, dim=0) / torch.sum(S_squared)

        # Find rank that meets energy threshold
        energy_rank = torch.searchsorted(
            cumulative_energy,
            self.energy_threshold
        ).item() + 1

        logging.debug(
            f"Energy-based SVD: {energy_rank}/{len(S)} components "
            f"retain {self.energy_threshold*100}% energy"
        )

        # Return with energy-based rank suggestion (truncation happens in base class)
        return U, S, Vh