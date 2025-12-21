"""
Base classes for tensor decomposition operations.

Provides abstract base class and common functionality for SVD, QR, and other
decomposition methods. Consolidates shared logic like shape handling, dtype
conversion, and statistics calculation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from enum import Enum

import torch

from ..types import MIN_SINGULAR_VALUE, DecompositionResult, SVDStats


class DecompositionMethod(Enum):
    """Enumeration of supported decomposition methods."""
    SVD = "svd"
    QR = "qr"
    RSVD = "rsvd"  # Randomized SVD
    ENERGY_RSVD = "energy_rsvd"  # Energy-based randomized SVD


class SingularValueDistribution(Enum):
    """How to distribute singular values in decomposition."""
    SYMMETRIC = "symmetric"  # sqrt(S) in both U and Vh
    ASYMMETRIC = "asymmetric"  # All S in U matrix


class TensorDecomposer(ABC):
    """
    Abstract base class for tensor decomposition operations.

    Provides common functionality for all decomposition methods:
    - Shape validation and reshaping (2D, 3D, 4D tensors)
    - Dtype conversion and device management
    - Error handling with fallback strategies
    - Statistics calculation
    - Rank selection strategies

    Subclasses implement the actual decomposition algorithm.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        distribution: SingularValueDistribution = SingularValueDistribution.SYMMETRIC,
        return_statistics: bool = False
    ):
        """
        Initialize decomposer.

        Args:
            device: Device for computation ('cpu', 'cuda')
            dtype: Data type for computation (SVD requires float32)
            distribution: How to distribute singular values
            return_statistics: Whether to return statistics
        """
        self.device = device
        self.dtype = dtype
        self.distribution = distribution
        self.return_statistics = return_statistics

    def decompose(
        self,
        weight: torch.Tensor,
        target_rank: int,
        dynamic_method: Optional[str] = None,
        dynamic_param: Optional[float] = None,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[Dict[str, Any]]]:
        """
        Decompose weight tensor into low-rank representation.

        Main entry point that handles the full decomposition pipeline:
        1. Validate and prepare tensor
        2. Perform decomposition (subclass-specific)
        3. Distribute singular values
        4. Reshape and convert back
        5. Calculate statistics

        Args:
            weight: Weight tensor to decompose (2D, 3D, or 4D)
            target_rank: Target rank for decomposition
            dynamic_method: Dynamic rank selection method (optional)
            dynamic_param: Parameter for dynamic method
            scale: Scale factor for alpha calculation

        Returns:
            Tuple of (up, down, alpha, stats)
            - up: Up-projection tensor
            - down: Down-projection tensor
            - alpha: Alpha scaling factor
            - stats: Statistics dictionary (if return_statistics=True)

        Raises:
            ValueError: If tensor shape is invalid
            RuntimeError: If decomposition fails
        """
        # Validate and prepare
        original_shape, original_dtype, is_conv = self._validate_tensor(weight)

        # Reshape to 2D if needed
        weight_2d, out_size, in_size = self._reshape_to_2d(weight, is_conv)

        # Convert dtype and device
        weight_2d = self._prepare_for_computation(weight_2d)

        try:
            # Perform decomposition (subclass-specific)
            U, S, Vh = self._decompose_2d(weight_2d)

            # Determine actual rank
            new_rank, new_alpha = self._determine_rank(
                S, target_rank, dynamic_method, dynamic_param, scale
            )

            # Ensure rank doesn't exceed available
            rank = min(new_rank, S.size(0))

            # Truncate and distribute
            up, down = self._truncate_and_distribute(U, S, Vh, rank)

            # Reshape back to original format
            up, down = self._reshape_back(up, down, original_shape, is_conv, rank)

            # Convert back to CPU and original dtype
            up = up.to(device='cpu', dtype=original_dtype)
            down = down.to(device='cpu', dtype=original_dtype)

            # Calculate statistics
            stats = None
            if self.return_statistics:
                stats = self._calculate_statistics(S, rank, new_alpha)

            return up, down, new_alpha, stats

        except RuntimeError as e:
            # Handle decomposition failures
            return self._handle_decomposition_error(e, weight, target_rank, scale)

    @abstractmethod
    def _decompose_2d(
        self,
        weight_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the actual 2D decomposition.

        Subclasses implement the specific algorithm (SVD, QR, etc.).

        Args:
            weight_2d: 2D weight tensor prepared for decomposition

        Returns:
            Tuple of (U, S, Vh) components

        Raises:
            RuntimeError: If decomposition fails
        """
        pass

    def _validate_tensor(
        self,
        weight: torch.Tensor
    ) -> Tuple[torch.Size, torch.dtype, bool]:
        """
        Validate tensor and extract metadata.

        Args:
            weight: Tensor to validate

        Returns:
            Tuple of (original_shape, original_dtype, is_conv)

        Raises:
            ValueError: If tensor shape is invalid
        """
        original_shape = weight.shape
        original_dtype = weight.dtype

        # Check dimensions
        ndim = len(original_shape)
        if ndim not in (2, 3, 4):
            raise ValueError(
                f"Tensor must be 2D, 3D, or 4D for decomposition. "
                f"Got shape: {original_shape}"
            )

        is_conv = ndim == 4

        # Validate conv layers have square kernels
        if is_conv and original_shape[2] != original_shape[3]:
            logging.warning(
                f"Conv layer has non-square kernel: {original_shape[2]}x{original_shape[3]}. "
                "This may cause issues."
            )

        return original_shape, original_dtype, is_conv

    def _reshape_to_2d(
        self,
        weight: torch.Tensor,
        is_conv: bool
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Reshape tensor to 2D for decomposition.

        Args:
            weight: Tensor to reshape
            is_conv: Whether tensor is convolutional

        Returns:
            Tuple of (weight_2d, out_size, in_size)
        """
        if is_conv:
            out_size, in_size, kernel_size, _ = weight.shape
            weight_2d = weight.reshape(out_size, -1)
        else:
            weight_2d = weight
            out_size, in_size = weight_2d.shape

        return weight_2d, out_size, in_size

    def _prepare_for_computation(self, weight_2d: torch.Tensor) -> torch.Tensor:
        """
        Prepare tensor for decomposition (dtype and device conversion).

        Args:
            weight_2d: 2D tensor

        Returns:
            Tensor on correct device with correct dtype
        """
        return weight_2d.to(dtype=self.dtype, device=self.device)

    def _determine_rank(
        self,
        S: torch.Tensor,
        target_rank: int,
        dynamic_method: Optional[str],
        dynamic_param: Optional[float],
        scale: float
    ) -> Tuple[int, float]:
        """
        Determine actual rank to use based on singular values.

        Args:
            S: Singular values tensor
            target_rank: Target rank
            dynamic_method: Dynamic selection method
            dynamic_param: Parameter for dynamic method
            scale: Scale factor

        Returns:
            Tuple of (new_rank, new_alpha)
        """
        if dynamic_method is None:
            new_rank = target_rank
            new_alpha = float(scale * new_rank)
            return new_rank, new_alpha

        # Check for zero matrix
        if S[0] <= MIN_SINGULAR_VALUE:
            logging.warning("Matrix is numerically zero (max singular value < 1e-6)")
            return 1, float(scale)

        # Dynamic rank selection
        if dynamic_method == "sv_ratio":
            new_rank = self._rank_by_ratio(S, dynamic_param)
        elif dynamic_method == "sv_cumulative":
            new_rank = self._rank_by_cumulative(S, dynamic_param)
        elif dynamic_method == "sv_fro":
            new_rank = self._rank_by_frobenius(S, dynamic_param)
        else:
            raise ValueError(
                f"Unknown dynamic method: {dynamic_method}. "
                "Supported: sv_ratio, sv_cumulative, sv_fro"
            )

        # Cap rank at target
        new_rank = min(max(new_rank, 1), target_rank)
        new_alpha = float(scale * new_rank)

        return new_rank, new_alpha

    def _rank_by_ratio(self, S: torch.Tensor, ratio: float) -> int:
        """Select rank based on singular value ratio."""
        max_sv = S[0]
        min_sv = max_sv / ratio
        return max(torch.sum(S > min_sv).item(), 1)

    def _rank_by_cumulative(self, S: torch.Tensor, threshold: float) -> int:
        """Select rank based on cumulative sum of singular values."""
        s_cumulative = torch.cumsum(S, dim=0)
        s_total = s_cumulative[-1]
        target_value = threshold * s_total
        return max(torch.searchsorted(s_cumulative, target_value).item() + 1, 1)

    def _rank_by_frobenius(self, S: torch.Tensor, threshold: float) -> int:
        """Select rank based on Frobenius norm retention."""
        S_squared = S.pow(2)
        s_cumulative = torch.cumsum(S_squared, dim=0)
        s_total = s_cumulative[-1]
        target_value = (threshold ** 2) * s_total
        rank = torch.searchsorted(s_cumulative, target_value).item() + 1
        return min(max(rank, 1), len(S) - 1)

    def _truncate_and_distribute(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truncate matrices and distribute singular values.

        Args:
            U: Left singular vectors
            S: Singular values
            Vh: Right singular vectors
            rank: Rank to truncate to

        Returns:
            Tuple of (up, down) tensors
        """
        # Truncate
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vh_truncated = Vh[:rank, :]

        # Distribute singular values
        if self.distribution == SingularValueDistribution.SYMMETRIC:
            # Symmetric: sqrt(S) in both matrices
            S_sqrt = torch.sqrt(S_truncated)
            up = U_truncated @ torch.diag(S_sqrt)
            down = torch.diag(S_sqrt) @ Vh_truncated
        else:
            # Asymmetric: all S in up matrix
            up = U_truncated @ torch.diag(S_truncated)
            down = Vh_truncated

        return up, down

    def _reshape_back(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        original_shape: torch.Size,
        is_conv: bool,
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape tensors back to original format.

        Args:
            up: Up-projection tensor
            down: Down-projection tensor
            original_shape: Original tensor shape
            is_conv: Whether original was convolutional
            rank: Decomposition rank

        Returns:
            Tuple of reshaped (up, down) tensors
        """
        if is_conv:
            out_size, in_size, kernel_h, kernel_w = original_shape
            down = down.reshape(rank, in_size, kernel_h, kernel_w)
            up = up.reshape(out_size, rank, 1, 1)

        return up, down

    def _calculate_statistics(
        self,
        S: torch.Tensor,
        rank: int,
        alpha: float
    ) -> Dict[str, Any]:
        """
        Calculate decomposition statistics.

        Args:
            S: Singular values
            rank: Used rank
            alpha: Alpha value

        Returns:
            Statistics dictionary
        """
        s_sum = torch.sum(torch.abs(S))
        s_rank = torch.sum(torch.abs(S[:rank]))

        S_squared = S.pow(2)
        s_fro = torch.sqrt(torch.sum(S_squared))
        s_red_fro = torch.sqrt(torch.sum(S_squared[:rank]))

        return {
            'new_rank': rank,
            'new_alpha': alpha,
            'sum_retained': float(s_rank / s_sum) if s_sum > 0 else 0.0,
            'fro_retained': float(s_red_fro / s_fro) if s_fro > 0 else 0.0,
            'max_ratio': float(S[0] / S[rank]) if rank < len(S) and S[rank] > 0 else float('inf')
        }

    def _handle_decomposition_error(
        self,
        error: Exception,
        weight: torch.Tensor,
        target_rank: int,
        scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[Dict]]:
        """
        Handle decomposition errors with fallback strategies.

        Args:
            error: The exception that occurred
            weight: Original weight tensor
            target_rank: Target rank
            scale: Scale factor

        Returns:
            Fallback decomposition result

        Raises:
            RuntimeError: If all fallback strategies fail
        """
        logging.error(f"Decomposition failed: {error}")

        # Try CPU fallback if we were on GPU
        if self.device != "cpu":
            logging.info("Attempting CPU fallback...")
            try:
                old_device = self.device
                self.device = "cpu"
                result = self.decompose(weight, target_rank, scale=scale)
                self.device = old_device  # Restore
                logging.info("CPU fallback succeeded")
                return result
            except Exception as e:
                logging.error(f"CPU fallback also failed: {e}")

        # If all else fails, raise with helpful message
        raise RuntimeError(
            f"Decomposition failed for tensor shape {weight.shape}: {error}. "
            f"Attempted fallback strategies also failed. "
            f"This may indicate a singular or ill-conditioned matrix."
        ) from error
