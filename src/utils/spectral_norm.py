"""
Spectral norm regularization for LoRA merging.

Spectral norm is the maximum singular value of a matrix, which represents its
Lipschitz constant. This module provides functions to compute and apply spectral
norm regularization to LoRA weights, preventing any single layer from dominating
the merge due to large weight magnitudes.

Inspired by the dare_nodes.py implementation.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _L2Normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a vector to unit L2 norm.

    Args:
        v: Input tensor (vector)
        eps: Small epsilon to prevent division by zero

    Returns:
        Normalized tensor with unit L2 norm
    """
    return v / (torch.norm(v) + eps)


def spectral_norm(
    W: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    num_iter: int = 10,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the spectral norm (maximum singular value) of a weight matrix using power iteration.

    The spectral norm represents the Lipschitz constant of the linear transformation
    defined by the matrix W. This is useful for understanding how much a layer can
    amplify or attenuate signals.

    Power iteration procedure:
    1. Start with random vector u
    2. Compute v = normalize(u @ W)
    3. Compute u = normalize(v @ W.T)
    4. Repeat steps 2-3 for num_iter iterations
    5. Spectral norm ≈ u @ W @ v.T

    Args:
        W: Weight matrix (shape: [out_features, in_features] or any 2D+ tensor)
        u: Initial random vector (optional, will be generated if None)
        num_iter: Number of power iterations (default: 10, more = more accurate)
        device: Device to run computation on (default: same as W)

    Returns:
        Tuple of (spectral_norm_value, final_u_vector)
        - spectral_norm_value: Maximum singular value (scalar tensor)
        - final_u_vector: Converged left singular vector

    Raises:
        ValueError: If num_iter < 1

    Example:
        >>> W = torch.randn(512, 256)
        >>> sn, u = spectral_norm(W, num_iter=20)
        >>> print(f"Spectral norm: {sn.item():.4f}")
    """
    if not num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")

    # Determine device
    if device is None:
        device = W.device

    # Initialize random vector if not provided
    if u is None:
        u = torch.randn(1, W.size(0), device=device, dtype=torch.float32)

    # Convert to float32 for numerical stability
    W_float = W.to(device).type(torch.float32)
    u = u.to(device).type(torch.float32)

    # Power iteration
    wdata = W_float.data
    for _ in range(num_iter):
        # Reshape weight to 2D matrix [out_features, in_features]
        wdata_2d = wdata.view(u.shape[-1], -1)

        # v = normalize(u @ W)
        v = _L2Normalize(torch.matmul(u, wdata_2d))

        # u = normalize(v @ W.T)
        u = _L2Normalize(torch.matmul(v, torch.transpose(wdata_2d, 0, 1)))

    # Compute spectral norm: sigma = u @ W @ v.T
    wdata_2d = wdata.view(u.shape[-1], -1)
    sigma = torch.sum(F.linear(u, torch.transpose(wdata_2d, 0, 1)) * v)

    return sigma, u


def apply_spectral_norm(
    lora_patches: Dict[str, torch.Tensor],
    scale: float,
    num_iter: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply spectral norm regularization to LoRA weights.

    This function:
    1. Computes the spectral norm (max singular value) for each weight tensor
    2. Finds the maximum spectral norm across all layers
    3. Scales all weights so that max_spectral_norm = target_scale

    This prevents any single layer from dominating the merge due to large weight
    magnitudes, leading to more stable and balanced merges.

    Args:
        lora_patches: Dictionary of LoRA weights {layer_key: weight_tensor}
                     Alpha keys (containing "alpha") are skipped
        scale: Target maximum spectral norm. Common values:
               - 0.1-0.5: Conservative, prevents overfitting
               - 1.0: Neutral scaling
               - 2.0-5.0: Allows stronger effects
        num_iter: Power iteration count for spectral norm computation (default: 10)
        device: Device to run computation on (optional)

    Returns:
        Dictionary of scaled LoRA weights with same structure as input

    Example:
        >>> lora = {"layer1.weight": torch.randn(512, 256),
        ...         "layer2.weight": torch.randn(256, 128),
        ...         "layer1.alpha": torch.tensor(32.0)}
        >>> regularized = apply_spectral_norm(lora, scale=1.0)
        >>> # All weight layers now have spectral norm <= 1.0

    Note:
        Alpha values are preserved unchanged, as they represent LoRA rank scaling
        factors, not learnable weights.
    """
    # Compute spectral norms for all non-alpha layers
    spectral_norms = []
    for key in lora_patches.keys():
        # Skip alpha values (LoRA scaling factors)
        if "alpha" in key.lower():
            continue

        weight = lora_patches[key]

        # Skip 0D/1D tensors (scalars, biases)
        if weight.ndim < 2:
            logging.debug(f"Skipping spectral norm for {key} (ndim={weight.ndim})")
            continue

        # Compute spectral norm
        try:
            sn, _ = spectral_norm(weight, num_iter=num_iter, device=device)
            spectral_norms.append(sn.cpu().item())
        except Exception as e:
            logging.warning(f"Failed to compute spectral norm for {key}: {e}")
            continue

    if not spectral_norms:
        logging.warning("No valid spectral norms computed, returning original weights")
        return lora_patches

    # Find maximum spectral norm across all layers
    max_sn = max(spectral_norms)

    if max_sn < 1e-8:
        logging.warning(f"Maximum spectral norm is very small ({max_sn}), skipping regularization")
        return lora_patches

    # Compute scaling factor
    scale_factor = scale / max_sn

    logging.info(
        f"Spectral norm regularization: max_sn={max_sn:.4f}, "
        f"target={scale:.4f}, scale_factor={scale_factor:.4f}"
    )

    # Apply scaling to all weight tensors
    regularized_patches = {}
    for key in lora_patches.keys():
        if "alpha" in key.lower():
            # Preserve alpha values unchanged
            regularized_patches[key] = lora_patches[key]
        else:
            # Scale weight tensors
            regularized_patches[key] = lora_patches[key] * scale_factor

    return regularized_patches


def compute_spectral_norms(
    lora_patches: Dict[str, torch.Tensor],
    num_iter: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Compute spectral norms for all layers (for analysis/debugging).

    Args:
        lora_patches: Dictionary of LoRA weights
        num_iter: Power iteration count
        device: Device to run on

    Returns:
        Dictionary mapping layer names to their spectral norms

    Example:
        >>> norms = compute_spectral_norms(lora_patches)
        >>> print(f"Layer spectral norms: {norms}")
    """
    norms = {}

    for key, weight in lora_patches.items():
        if "alpha" in key.lower() or weight.ndim < 2:
            continue

        try:
            sn, _ = spectral_norm(weight, num_iter=num_iter, device=device)
            norms[key] = sn.cpu().item()
        except Exception as e:
            logging.debug(f"Failed to compute spectral norm for {key}: {e}")

    return norms
