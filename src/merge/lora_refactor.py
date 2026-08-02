"""Refactor a merged full delta back into a LoRA (up, down, alpha).

Uses randomized SVD (`torch.svd_lowrank`) with energy-based dynamic rank instead
of a full `torch.linalg.svd`. Full SVD computes the entire spectrum plus a large
cuSOLVER/LAPACK workspace (~190 MB and ~2.9 s for a 3072x3072 layer here); the
randomized path computes only the top ~rank components (~6 MB, ~4 ms). Run across
the merge thread-pool under a resident diffusion model, the full-SVD path is both
a VRAM hog and the fragile cuSOLVER driver that crashes under pressure — the
randomized path avoids both.

Pure torch, no intra-package imports, so it stays loadable by file path in the
repo's broken test env (mirrors gta.py)."""
from typing import Callable, Optional, Tuple

import torch

Calibrator = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def merged_delta_to_lora(delta: torch.Tensor, target_rank: int, *,
                         energy: Optional[float] = 0.99, device="cpu",
                         dtype: torch.dtype = torch.float32,
                         oversample: int = 8, niter: int = 2,
                         calibrate: Optional[Calibrator] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Decompose ``delta`` into ``(up, down, alpha)`` via randomized SVD.

    The reconstruction convention matches the rest of the node: the effective
    LoRA delta is ``(alpha / rank) * (up @ down)``, and with ``alpha == rank``
    that is exactly ``up @ down`` (the rank-``r`` approximation of ``delta``).

    Args:
        delta: merged full delta, 2D ``[out, in]`` (4D conv is flattened).
        target_rank: hard cap on the output rank (e.g. max input LoRA rank).
        energy: cumulative singular-value energy to retain, in ``(0, 1]``. The
            rank is the smallest ``r <= target_rank`` reaching that fraction of
            ``||delta||_F^2``. ``None`` uses ``target_rank`` directly.
        oversample: extra components sampled for randomized-SVD accuracy.
        niter: power iterations (1-2 is plenty for low-rank deltas).
        calibrate: optional ``(U_r, S_r, V_r) -> gamma`` hook that rescales the
            retained singular values (see src/merge/svc.py). Applied *after* the
            rank is chosen, so it changes magnitudes only -- never the rank or
            the singular vectors. ``None`` reproduces the uncalibrated path
            exactly.
    """
    orig_shape = delta.shape
    orig_dtype = delta.dtype
    is_conv = delta.dim() == 4
    if is_conv:
        W = delta.reshape(orig_shape[0], -1)
    else:
        W = delta
    W = W.to(device=device, dtype=torch.float32)
    out_f, in_f = W.shape

    max_rank = min(out_f, in_f)
    target_rank = max(1, min(int(target_rank), max_rank))

    total_energy = W.pow(2).sum()
    if total_energy <= 1e-12:  # all-zero delta: return a rank-1 zero LoRA
        up = torch.zeros(out_f, 1, dtype=orig_dtype)
        down = torch.zeros(1, in_f, dtype=orig_dtype)
        if is_conv:
            up = up.reshape(out_f, 1, 1, 1)
            down = down.reshape(1, orig_shape[1], orig_shape[2], orig_shape[3])
        return up, down, 1.0

    q = min(target_rank + oversample, max_rank)
    U, S, V = torch.svd_lowrank(W, q=q, niter=niter)  # S descending

    if energy is not None:
        cum = torch.cumsum(S.pow(2), dim=0) / total_energy
        thresh = torch.tensor(float(energy), device=cum.device, dtype=cum.dtype)
        r = int(torch.searchsorted(cum, thresh).item()) + 1
        r = max(1, min(r, target_rank, S.numel()))
    else:
        r = min(target_rank, S.numel())

    S_r = S[:r]
    if calibrate is not None:
        gamma = calibrate(U[:, :r], S_r, V[:, :r])
        S_r = S_r * gamma.to(device=S_r.device, dtype=S_r.dtype)
    s_sqrt = S_r.clamp_min(0).sqrt()
    up = U[:, :r] * s_sqrt.unsqueeze(0)          # [out, r]
    down = (V[:, :r] * s_sqrt.unsqueeze(0)).T    # [r, in]
    alpha = float(r)                             # scale = 1 convention

    if is_conv:
        up = up.reshape(out_f, r, 1, 1)
        down = down.reshape(r, orig_shape[1], orig_shape[2], orig_shape[3])
    up = up.to(device="cpu", dtype=orig_dtype)
    down = down.to(device="cpu", dtype=orig_dtype)
    return up, down, alpha
