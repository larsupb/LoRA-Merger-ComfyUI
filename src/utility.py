import logging
import os
from typing import Tuple, Dict, Optional

import torch
from PIL import ImageFont

from .types import LORA_TENSORS, DeviceType, DtypeType, MIN_SINGULAR_VALUE

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")

def map_device(device: DeviceType, dtype: DtypeType) -> Tuple[torch.device, torch.dtype]:
    """
    Convert device and dtype from string representation to torch objects.

    Args:
        device: Device specification as string or torch.device
        dtype: Data type specification as string or torch.dtype

    Returns:
        Tuple of (torch.device, torch.dtype)
    """
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return device, dtype


def find_network_dim(lora_sd: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Find the network dimension (rank) of a LoRA from its state dict.

    Args:
        lora_sd: LoRA state dictionary

    Returns:
        Network dimension (rank) or None if not found
    """
    network_dim = None
    for key, value in lora_sd.items():
        if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
            network_dim = value.size()[0]
    return network_dim


def adjust_tensor_dims(
    ups_downs_alphas: Dict[str, LORA_TENSORS],
    apply_svd: bool = False,
    svd_rank: int = -1,
    method: str = 'rSVD'
) -> Dict[str, LORA_TENSORS]:
    """
    Checks if tensor dimensions and eventually aligns them to the first tensor with SVD or QR.

    Args:
        ups_downs_alphas: Dictionary mapping LoRA names to (up, down, alpha) tuples
        apply_svd: Whether to apply SVD/QR for dimension adjustment
        svd_rank: Target rank for adjustment. If -1, uses the rank of the first tensor
        method: Method to use for dimension adjustment: 'rSVD', 'SVD', or 'QR'

    Returns:
        Dictionary with aligned tensor dimensions

    Raises:
        ValueError: If dimensions don't match and apply_svd is False
    """
    up_0, up_1, _ = next(iter(ups_downs_alphas.values()))
    target_rank = up_0.shape[1] if svd_rank == -1 else svd_rank
    logging.debug(f"adjust_tensor_dims: Input svd_rank={svd_rank}, up_0 shape={up_0.shape}, computed target_rank={target_rank}, method={method}, apply_svd={apply_svd}")

    out = {}
    for lora_name, (up, down, alpha) in ups_downs_alphas.items():
        logging.debug(f"adjust_tensor_dims: Processing '{lora_name}' - up shape={up.shape}, down shape={down.shape}, alpha={alpha}")
        if up.shape[1] != target_rank:
            if not apply_svd:
                raise ValueError(f"LoRA up tensors have different shapes: {up.shape} vs {up_0.shape}. "
                                 f"Turn on apply_svd to True to resize them.")
            logging.debug(f"adjust_tensor_dims: Resizing '{lora_name}' from rank {up.shape[1]} to {target_rank} using method '{method}'")
            original_dtype = up.dtype

            # Choose resize method based on 'method' parameter
            if method == 'rSVD':
                down, up = resize_lora_rank_rsvd(
                    down.to(dtype=torch.float32),
                    up.to(dtype=torch.float32),
                    target_rank)
            elif method == 'energy_rSVD':
                down, up = resize_lora_rank_energy_rsvd(
                    down.to(dtype=torch.float32),
                    up.to(dtype=torch.float32),
                    target_rank)
            else:  # default to 'svd'
                down, up = resize_lora_rank(
                    down.to(dtype=torch.float32),
                    up.to(dtype=torch.float32),
                    target_rank)

            down = down.to(device="cpu", dtype=original_dtype)
            up = up.to(device="cpu", dtype=original_dtype)
            logging.debug(f"adjust_tensor_dims: After resize '{lora_name}' - up shape={up.shape}, down shape={down.shape}")
        else:
            logging.debug(f"adjust_tensor_dims: '{lora_name}' already at target rank, no resize needed")
        out[lora_name] = (up, down, alpha)
    return out


def perform_lora_svd(
    weight: torch.Tensor,
    target_rank: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    dynamic_method: str = None,
    dynamic_param: float = None,
    scale: float = 1.0,
    distribute_singular_values: bool = True,
    return_statistics: bool = False
) -> Tuple:
    """
    Unified SVD function for LoRA rank resizing.

    This function performs Singular Value Decomposition on a weight matrix and returns
    low-rank decomposition (up, down, alpha) with optional statistics.

    Args:
        weight: Full weight matrix to decompose (can be 2D, 3D, or 4D for conv layers)
        target_rank: Target rank for decomposition
        device: Device for computation ('cpu' or 'cuda')
        dtype: Data type for computation (default: float32, required for SVD)
        dynamic_method: Method for dynamic rank selection:
            - None: Use target_rank directly
            - 'sv_ratio': Select rank based on singular value ratio
            - 'sv_cumulative': Select rank based on cumulative sum of singular values
            - 'sv_fro': Select rank based on Frobenius norm
        dynamic_param: Parameter for dynamic method (interpretation depends on method)
        scale: Scale factor for alpha calculation (default: 1.0)
        distribute_singular_values: If True, split sqrt(S) between U and Vh (symmetric);
                                    if False, put all S in U (asymmetric, used by lora_resize)
        return_statistics: If True, return dictionary with statistics

    Returns:
        If return_statistics is False:
            (up, down, alpha) - Tuple of tensors
        If return_statistics is True:
            (up, down, alpha, stats_dict) - Tuple with statistics dictionary containing:
                - 'new_rank': Actually used rank
                - 'new_alpha': Calculated alpha value
                - 'sum_retained': Fraction of singular values sum retained
                - 'fro_retained': Fraction of Frobenius norm retained
                - 'max_ratio': Ratio between largest and rank-th singular value
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    is_conv = len(original_shape) == 4

    # Reshape conv layers to 2D for SVD
    if is_conv:
        out_size, in_size, kernel_size, _ = original_shape
        weight_2d = weight.reshape(out_size, -1)
    else:
        weight_2d = weight
        out_size, in_size = weight_2d.shape

    # SVD requires float32
    weight_2d = weight_2d.to(dtype=torch.float32, device=device)

    # Perform SVD (this operation itself cannot be interrupted)
    U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)

    # Determine actual rank to use
    if dynamic_method is None:
        new_rank = target_rank
        new_alpha = float(scale * new_rank)
    else:
        # Dynamic rank selection based on singular values
        MIN_SV = 1e-6

        if S[0] <= MIN_SV:  # Zero matrix
            new_rank = 1
            new_alpha = float(scale * new_rank)
        elif dynamic_method == "sv_ratio":
            # Select rank based on singular value ratio
            max_sv = S[0]
            min_sv = max_sv / dynamic_param
            new_rank = max(torch.sum(S > min_sv).item(), 1)
            new_alpha = float(scale * new_rank)
        elif dynamic_method == "sv_cumulative":
            # Select rank based on cumulative sum
            new_rank = index_sv_cumulative(S, dynamic_param)
            new_rank = max(new_rank, 1)
            new_alpha = float(scale * new_rank)
        elif dynamic_method == "sv_fro":
            # Select rank based on Frobenius norm
            new_rank = index_sv_fro(S, dynamic_param)
            new_rank = min(max(new_rank, 1), len(S) - 1)
            new_alpha = float(scale * new_rank)
        else:
            raise ValueError(f"Unknown dynamic method: {dynamic_method}")

        # Cap rank at target_rank
        if new_rank > target_rank:
            new_rank = target_rank
            new_alpha = float(scale * new_rank)

    # Ensure rank doesn't exceed available singular values
    rank = min(new_rank, S.size(0))

    # Truncate matrices
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    Vh_truncated = Vh[:rank, :]

    # Distribute singular values
    if distribute_singular_values:
        # Symmetric: sqrt(S) in both matrices
        S_sqrt = torch.sqrt(S_truncated)
        up = U_truncated @ torch.diag(S_sqrt)
        down = torch.diag(S_sqrt) @ Vh_truncated
    else:
        # Asymmetric: all S in up matrix (used by lora_resize.py)
        up = U_truncated @ torch.diag(S_truncated)
        down = Vh_truncated

    # Reshape back to original format
    if is_conv:
        down = down.reshape(rank, in_size, original_shape[2], original_shape[3])
        up = up.reshape(out_size, rank, 1, 1)

    # Convert back to CPU and original dtype
    up = up.to(device='cpu', dtype=original_dtype)
    down = down.to(device='cpu', dtype=original_dtype)

    # Calculate statistics if requested
    if return_statistics:
        s_sum = torch.sum(torch.abs(S))
        s_rank = torch.sum(torch.abs(S[:rank]))
        S_squared = S.pow(2)
        s_fro = torch.sqrt(torch.sum(S_squared))
        s_red_fro = torch.sqrt(torch.sum(S_squared[:rank]))

        stats = {
            'new_rank': rank,
            'new_alpha': new_alpha,
            'sum_retained': float(s_rank / s_sum),
            'fro_retained': float(s_red_fro / s_fro),
            'max_ratio': float(S[0] / S[rank]) if rank < len(S) else float('inf')
        }
        return up, down, new_alpha, stats

    return up, down, new_alpha


def perform_lora_qr(
    weight: torch.Tensor,
    target_rank: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    scale: float = 1.0,
    distribute_singular_values: bool = True,
    return_statistics: bool = False
) -> Tuple:
    """
    QR-based LoRA rank reduction - faster alternative to SVD.

    This function performs QR decomposition on a weight matrix and returns
    low-rank decomposition (up, down, alpha). QR is typically 2-5x faster than SVD
    but provides less optimal approximation. Best used for fixed-rank scenarios
    where speed is prioritized over optimality.

    Args:
        weight: Full weight matrix to decompose (can be 2D, 3D, or 4D for conv layers)
        target_rank: Target rank for decomposition (fixed, no dynamic selection)
        device: Device for computation ('cpu' or 'cuda')
        dtype: Data type for computation (default: float32)
        scale: Scale factor for alpha calculation (default: 1.0)
        distribute_singular_values: If True, split R values between up and down (symmetric);
                                    if False, put all R in up (asymmetric)
        return_statistics: If True, return dictionary with basic statistics

    Returns:
        If return_statistics is False:
            (up, down, alpha) - Tuple of tensors
        If return_statistics is True:
            (up, down, alpha, stats_dict) - Tuple with statistics dictionary containing:
                - 'new_rank': Actually used rank
                - 'new_alpha': Calculated alpha value
                - 'sum_retained': Not available for QR (returns 1.0)
                - 'fro_retained': Approximate Frobenius norm retention
                - 'max_ratio': Not available for QR (returns 1.0)

    Note: Unlike SVD, QR decomposition doesn't support dynamic rank selection based on
    singular values. Use perform_lora_svd() if you need sv_ratio, sv_cumulative, or sv_fro.
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    is_conv = len(original_shape) == 4

    # Reshape conv layers to 2D for QR
    if is_conv:
        out_size, in_size, kernel_size, _ = original_shape
        weight_2d = weight.reshape(out_size, -1)
    else:
        weight_2d = weight
        out_size, in_size = weight_2d.shape

    # QR works with float32
    weight_2d = weight_2d.to(dtype=torch.float32, device=device)

    # Perform QR decomposition: W = Q @ R where Q is orthogonal
    # For low-rank approximation: W ≈ Q[:, :r] @ R[:r, :]
    # We decompose W^T to get the right shape orientation
    Q, R = torch.linalg.qr(weight_2d.T, mode='reduced')  # Q: (in_size, min(out,in)), R: (min(out,in), out_size)

    # Ensure rank doesn't exceed available dimensions
    max_rank = min(out_size, in_size)
    rank = min(target_rank, max_rank)
    new_alpha = float(scale * rank)

    # Truncate matrices to target rank
    Q_truncated = Q[:, :rank]  # (in_size, rank)
    R_truncated = R[:rank, :]  # (rank, out_size)

    # Distribute values between up and down matrices
    if distribute_singular_values:
        # Symmetric: Split the R diagonal values as sqrt
        # Extract diagonal of R (or use scaling)
        R_diag = torch.diagonal(R_truncated[:, :rank])
        R_sqrt = torch.sqrt(torch.abs(R_diag) + 1e-8) * torch.sign(R_diag)

        # Scale Q and R appropriately
        down = Q_truncated @ torch.diag(R_sqrt)  # (in_size, rank)
        up = torch.diag(R_sqrt) @ R_truncated  # (rank, out_size)
        up = up.T  # (out_size, rank)
    else:
        # Asymmetric: All R information in up matrix
        down = Q_truncated  # (in_size, rank)
        up = R_truncated.T  # (out_size, rank)

    # Reshape back to original format if conv layer
    if is_conv:
        down = down.T.reshape(rank, in_size, original_shape[2], original_shape[3])
        up = up.reshape(out_size, rank, 1, 1)
    else:
        # For linear layers: down is (rank, in_size), up is (out_size, rank)
        down = down.T

    # Convert back to CPU and original dtype
    up = up.to(device='cpu', dtype=original_dtype)
    down = down.to(device='cpu', dtype=original_dtype)

    # Calculate statistics if requested
    if return_statistics:
        # For QR, we can't directly compute singular value statistics
        # Provide approximate Frobenius norm retention
        if is_conv:
            weight_fro = torch.norm(weight_2d, p='fro')
            # Reconstruct approximation
            down_2d = down.reshape(rank, -1)
            up_2d = up.reshape(out_size, rank)
            approx = up_2d @ down_2d
            approx_fro = torch.norm(approx, p='fro')
            fro_retained = float(approx_fro / (weight_fro + 1e-8))
        else:
            weight_fro = torch.norm(weight_2d, p='fro')
            approx = up @ down
            approx_fro = torch.norm(approx, p='fro')
            fro_retained = float(approx_fro / (weight_fro + 1e-8))

        stats = {
            'new_rank': rank,
            'new_alpha': new_alpha,
            'sum_retained': 1.0,  # Not available for QR
            'fro_retained': fro_retained,
            'max_ratio': 1.0  # Not available for QR
        }
        return up, down, new_alpha, stats

    return up, down, new_alpha


def resize_lora_rank(down: torch.Tensor, up: torch.Tensor, new_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize LoRA weights to a new rank using SVD.

    This is a wrapper around perform_lora_svd() that maintains backward compatibility
    with the original function signature.

    Args:
        down (torch.Tensor): The low-rank down matrix.
        up (torch.Tensor): The low-rank up matrix.
        new_dim (int): The desired rank for the LoRA weights.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The resized low-rank matrices.
    """
    # Compute the full LoRA matrix
    try:
        if down.dim() > 2:
            down_ = down.squeeze(-2, -1)  # Ensure down is 2D
        else:
            down_ = down
        if up.dim() > 2:
            up_ = up.squeeze(-2, -1)
        else:
            up_ = up
        W = up_ @ down_  # Shape: (out_features, in_features)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to compute full LoRA matrix: {e}. "
                           "Ensure that 'up' and 'down' tensors are compatible for matrix multiplication.")

    # Use unified SVD function with symmetric distribution
    up_new, down_new, _ = perform_lora_svd(
        weight=W,
        target_rank=new_dim,
        device=up.device.type if up.device.type != 'cpu' else 'cpu',
        dtype=up.dtype,
        distribute_singular_values=True,  # Use symmetric distribution
        return_statistics=False
    )

    # Handle padding if needed (new_dim > actual_rank)
    actual_rank = up_new.shape[1]
    if new_dim > actual_rank:
        pad_up = torch.zeros((up.shape[0], new_dim - actual_rank), dtype=up.dtype, device='cpu')
        pad_down = torch.zeros((new_dim - actual_rank, down.shape[1]), dtype=down.dtype, device='cpu')
        up_new = torch.cat([up_new, pad_up], dim=1)
        down_new = torch.cat([down_new, pad_down], dim=0)

    # Restore original dimensionality if needed (for conv layers)
    while down.dim() > down_new.dim():
        down_new = down_new.unsqueeze(-1)
    while up.dim() > up_new.dim():
        up_new = up_new.unsqueeze(-1)

    return down_new, up_new


def resize_lora_rank_energy_rsvd(
    down: torch.Tensor,
    up: torch.Tensor,
    new_dim: int,
    energy_keep_ratio: float = 1.5,
    niter: int = 1,
    oversample: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid LoRA rank reduction:
    1) Energy-based rank pruning
    2) Optional randomized SVD refinement

    Fast, stable, DiT-safe.
    """

    # ---- shape handling (SAFE) ----
    up_shape = up.shape
    down_shape = down.shape

    def to_2d(t):
        if t.dim() == 4 and t.shape[-1] == 1 and t.shape[-2] == 1:
            return t[..., 0, 0]
        if t.dim() == 2:
            return t
        raise ValueError(f"Unsupported LoRA shape: {t.shape}")

    up_ = to_2d(up)
    down_ = to_2d(down)

    r0 = down_.shape[0]
    logging.debug(f"energy_rSVD: Input rank r0={r0}, target new_dim={new_dim}, up shape={up_.shape}, down shape={down_.shape}")

    # Handle upsampling case: when input rank < target rank
    if new_dim > r0:
        logging.debug(f"energy_rSVD: Upsampling case - new_dim > r0, using full SVD to upsample")
        # Reconstruct full weight matrix and perform SVD to upsample
        Wk = up_ @ down_

        # Use torch.svd_lowrank with target rank
        rank = new_dim
        q = min(rank + 2, min(Wk.shape))  # Small oversample for upsampling

        U, S, V = torch.svd_lowrank(Wk, q=q, niter=1)
        Vh = V.transpose(-2, -1)

        # Truncate to target rank
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Distribute singular values symmetrically
        S_sqrt = torch.sqrt(S)
        up_new = U * S_sqrt.unsqueeze(0)
        down_new = S_sqrt.unsqueeze(1) * Vh

        # Pad if necessary (when matrix dimensions are smaller than target)
        actual_rank = up_new.shape[1]
        if new_dim > actual_rank:
            pad_up = torch.zeros(
                (up_new.shape[0], new_dim - actual_rank),
                dtype=up.dtype,
                device=up_.device,
            )
            pad_down = torch.zeros(
                (new_dim - actual_rank, down_new.shape[1]),
                dtype=down.dtype,
                device=down_.device,
            )
            up_new = torch.cat([up_new, pad_up], dim=1)
            down_new = torch.cat([down_new, pad_down], dim=0)

        # Restore shapes
        if len(up_shape) == 4:
            up_new = up_new.unsqueeze(-1).unsqueeze(-1)
        if len(down_shape) == 4:
            down_new = down_new.unsqueeze(-1).unsqueeze(-1)

        logging.debug(f"energy_rSVD: Upsampled to rank {new_dim} - up_new={up_new.shape}, down_new={down_new.shape}")
        return down_new, up_new

    # Early exit when ranks match exactly
    if new_dim == r0:
        logging.debug(f"energy_rSVD: Early exit - new_dim == r0, returning original tensors")
        return down, up

    # ---- Phase 1: Energy pruning ----
    # Energy per rank component
    up_norm = torch.norm(up_, dim=0)
    down_norm = torch.norm(down_, dim=1)
    energy = up_norm * down_norm

    # Number of components to keep before SVD
    k = min(int(new_dim * energy_keep_ratio), r0)
    logging.debug(f"energy_rSVD: Calculated k={k} (energy_keep_ratio={energy_keep_ratio}, new_dim={new_dim}, r0={r0})")

    # Top-k indices
    idx = torch.topk(energy, k, largest=True).indices
    idx, _ = torch.sort(idx)

    up_k = up_[:, idx]
    down_k = down_[idx, :]
    logging.debug(f"energy_rSVD: After pruning - up_k shape={up_k.shape}, down_k shape={down_k.shape}")

    # ---- Early exit: pure pruning ----
    if k == new_dim:
        logging.debug(f"energy_rSVD: Early exit - k == new_dim, using pure pruning (no SVD refinement)")
        up_new = up_k
        down_new = down_k

    else:
        # ---- Phase 2: rSVD refinement ----
        logging.debug(f"energy_rSVD: Entering SVD refinement path (k={k} != new_dim={new_dim})")
        Wk = up_k @ down_k
        logging.debug(f"energy_rSVD: Wk shape={Wk.shape}")

        rank = new_dim
        q = min(rank + oversample, min(Wk.shape))
        logging.debug(f"energy_rSVD: SVD parameters - rank={rank}, q={q}, niter={niter}")

        U, S, V = torch.svd_lowrank(Wk, q=q, niter=niter)
        Vh = V.transpose(-2, -1)

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        logging.debug(f"energy_rSVD: After SVD slicing - U shape={U.shape}, S shape={S.shape}, Vh shape={Vh.shape}")

        S_sqrt = torch.sqrt(S)
        up_new = U * S_sqrt.unsqueeze(0)
        down_new = S_sqrt.unsqueeze(1) * Vh
        logging.debug(f"energy_rSVD: After scaling - up_new shape={up_new.shape}, down_new shape={down_new.shape}")

    # ---- restore shapes ----
    if len(up_shape) == 4:
        up_new = up_new.unsqueeze(-1).unsqueeze(-1)
    if len(down_shape) == 4:
        down_new = down_new.unsqueeze(-1).unsqueeze(-1)

    # ---- safety checks ----
    logging.debug(f"energy_rSVD: Final shapes before return - up_new={up_new.shape}, down_new={down_new.shape}, expected rank={new_dim}")
    assert up_new.shape[1] == new_dim, f"up_new rank mismatch: {up_new.shape[1]} != {new_dim}"
    assert down_new.shape[0] == new_dim, f"down_new rank mismatch: {down_new.shape[0]} != {new_dim}"

    return down_new, up_new

def resize_lora_rank_rsvd(
    down: torch.Tensor,
    up: torch.Tensor,
    new_dim: int,
    niter: int = 2,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize LoRA weights to a new rank using randomized SVD
    (fast, stable replacement for full SVD).

    Args:
        down (torch.Tensor): LoRA down matrix (r, in_features)
        up (torch.Tensor): LoRA up matrix (out_features, r)
        new_dim (int): Target LoRA rank
        niter (int): Power iterations (1–2 is usually enough)
        oversample (int): Oversampling for stability

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (down_new, up_new)
    """

    # --- Ensure 2D ---
    if down.dim() > 2:
        down_ = down.squeeze(-2, -1)
    else:
        down_ = down

    if up.dim() > 2:
        up_ = up.squeeze(-2, -1)
    else:
        up_ = up

    # --- Reconstruct full LoRA update ---
    try:
        W = up_ @ down_  # (out_features, in_features)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to compute full LoRA matrix: {e}. "
            "Ensure that 'up' and 'down' are compatible."
        )

    # --- Randomized SVD ---
    rank = min(new_dim, min(W.shape))
    q = min(rank + oversample, min(W.shape))

    # IMPORTANT: torch.svd_lowrank returns (U, S, V) where V is already transposed!
    # U: (out, q), S: (q,), V: (in, q)
    # To get Vh (which is V transposed), we need V.T
    U, S, V = torch.svd_lowrank(
        W,
        q=q,
        niter=niter,
    )

    # Truncate to target rank
    U = U[:, :rank]              # (out_features, rank)
    S = S[:rank]                 # (rank,)
    V = V[:, :rank]              # (in_features, rank)

    # --- Symmetric singular value distribution (LoRA-correct) ---
    S_sqrt = torch.sqrt(S)

    up_new = U * S_sqrt.unsqueeze(0)            # (out_features, rank)
    down_new = (V * S_sqrt.unsqueeze(0)).T      # (rank, in_features)

    # --- Move to CPU and convert dtype to match input ---
    up_new = up_new.to(device='cpu', dtype=up.dtype)
    down_new = down_new.to(device='cpu', dtype=down.dtype)

    # --- Padding if rank was limited by matrix size ---
    actual_rank = up_new.shape[1]
    if new_dim > actual_rank:
        pad_up = torch.zeros(
            (up_new.shape[0], new_dim - actual_rank),
            dtype=up.dtype,
            device='cpu',
        )
        pad_down = torch.zeros(
            (new_dim - actual_rank, down_new.shape[1]),
            dtype=down.dtype,
            device='cpu',
        )
        up_new = torch.cat([up_new, pad_up], dim=1)
        down_new = torch.cat([down_new, pad_down], dim=0)

    # --- Restore conv dimensions if needed ---
    while down.dim() > down_new.dim():
        down_new = down_new.unsqueeze(-1)
    while up.dim() > up_new.dim():
        up_new = up_new.unsqueeze(-1)

    return down_new, up_new


def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target ** 2)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def to_dtype(dtype):
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_mapping.get(dtype, torch.float32)
    return dtype


def load_font():
    # Load a font
    font_path = f"{FONTS_DIR}/ShareTechMono-Regular.ttf"
    try:
        title_font = ImageFont.truetype(font_path, size=48)
    except OSError:
        logging.warning(f"PM LoRABlockSampler: Font not found at {font_path}, using default font.")
        title_font = ImageFont.load_default()
    return title_font