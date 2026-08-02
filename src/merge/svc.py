"""Singular Value Calibration (SVC) for merged LoRA deltas.

When several LoRAs share aligned singular directions, a linear merge counts
those directions once per LoRA. The merged delta's singular values along shared
directions are inflated and the result is dominated by shared subspaces. SVC
measures the inflation per direction and divides it back out, rescaling singular
values while leaving the singular vectors untouched.

    "When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging"
    (ICML 2026), arXiv:2602.05536.

Two things make this cheap here. The merged response along direction u^r is
exactly sigma^r * (v^r)^T -- a row of diag(S) V^T, no matmul needed. And the
per-task response is built in factored form, ``U_r^T (scale * up @ down)``,
which never materializes a dense [out, in] per-task delta: peak is [r, in].

Pure torch, no intra-package imports, so it stays loadable by file path
(mirrors gta.py and lora_refactor.py)."""
from typing import Callable, List, Optional, Sequence, Tuple

import torch

# (up [out, r_i], down [r_i, in], scale) -- both 2D, conv already flattened.
Factor = Tuple[torch.Tensor, torch.Tensor, float]

# Below this response energy a task is considered absent from a direction and
# casts a neutral vote instead of a meaningless ratio. Expressed relative to the
# layer's leading singular value, so the threshold is scale-free.
_ABSENT_REL = 1e-6


def calibration_gammas(U_r: torch.Tensor, S_r: torch.Tensor, V_r: torch.Tensor,
                       factors: Sequence[Factor], alpha: float) -> torch.Tensor:
    """Per-direction calibration factors ``gamma`` for the retained spectrum.

    Args:
        U_r: left singular vectors of the merged delta, ``[out, r]``.
        S_r: retained singular values, ``[r]`` (descending).
        V_r: right singular vectors, ``[in, r]``.
        factors: one ``(up, down, scale)`` per task. ``scale`` must already
            include the LoRA's ``alpha/rank`` AND the user's merge strength, so
            that ``scale * up @ down`` is the task's contribution as the merge
            actually used it.
        alpha: floor in ``max(alpha, s_i)``. ``1.0`` suppresses only; smaller
            values also permit amplification.

    Returns:
        ``[r]`` tensor of strictly positive multipliers for ``S_r``.
    """
    k = len(factors)
    if k == 0:
        return torch.ones_like(S_r)

    work = torch.float32
    U_r = U_r.to(work)
    S_r_w = S_r.to(work)
    V_r = V_r.to(work)
    r = S_r_w.numel()

    # Merged response is diag(S) V^T, so the denominator-free part of the
    # numerator is just <v^r, a_i^r> scaled by sigma^r.
    Vt = V_r.T.contiguous()                       # [r, in]

    # One absolute "is this task present in this direction" threshold for the whole
    # layer, tied to the leading singular value so it is scale-free. A per-component
    # threshold would collapse to ~0 for trailing near-zero singular values and let
    # numerical noise through as a real ratio.
    floor = (_ABSENT_REL * S_r_w.max()).pow(2)

    denom_acc = torch.zeros(r, dtype=work, device=S_r_w.device)
    for up, down, scale in factors:
        up = up.to(device=U_r.device, dtype=work)
        down = down.to(device=U_r.device, dtype=work)
        # A_i = U_r^T (scale * up @ down), built right-to-left: [r, r_i] @ [r_i, in]
        A_i = (U_r.T @ up) @ down                 # [r, in]
        if scale != 1.0:
            A_i = A_i * float(scale)

        den = A_i.pow(2).sum(dim=1)               # ||a_i^r||^2, [r]
        num = S_r_w * (A_i * Vt).sum(dim=1)       # <a_m^r, a_i^r>, [r]
        del A_i

        # Absent tasks vote neutrally (s = 1): a task with no presence in a
        # direction must not drag gamma around for a direction it does not use.
        s_i = torch.where(den > floor, num / den.clamp_min(1e-30),
                          torch.ones_like(den))
        denom_acc += s_i.clamp_min(float(alpha))

    return k / denom_acc.clamp_min(1e-8)


# ---------------------------------------------------------------- modes
# Widget contract. "off" disables calibration entirely.
SVC_MODES = ("off", "balanced", "suppress_only")


def resolve_alpha(mode: str, n_tasks: int) -> Optional[float]:
    """Map a widget mode to the ``alpha`` floor, or None when SVC is off.

    ``balanced`` uses the paper default ``1/K``, which lets under-represented
    directions be amplified as well as suppressed. ``suppress_only`` uses
    ``alpha = 1``, which floors every term at 1 so ``gamma <= 1`` always.
    """
    if mode == "off":
        return None
    if mode == "suppress_only":
        return 1.0
    if mode == "balanced":
        return 1.0 / max(1, int(n_tasks))
    raise ValueError(f"unknown SVC mode {mode!r}; expected one of {SVC_MODES}")


# ---------------------------------------------------------------- factory
def _as_2d(t: torch.Tensor) -> torch.Tensor:
    """Flatten a 4D conv factor to 2D, matching merged_delta_to_lora's reshape of
    the merged delta (``up -> [out, r]``, ``down -> [r, in*kh*kw]``)."""
    return t.reshape(t.shape[0], -1) if t.dim() == 4 else t


def make_calibrator(factors: Sequence[Factor], mode: str
                    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                                           torch.Tensor]]:
    """Build the ``calibrate`` callback ``merged_delta_to_lora`` expects.

    Returns None when ``mode`` is ``"off"`` or there is nothing to calibrate, so
    callers can pass the result straight through without branching.
    """
    alpha = resolve_alpha(mode, len(factors))
    if alpha is None or not factors:
        return None

    flat: List[Factor] = [(_as_2d(u), _as_2d(d), float(s)) for u, d, s in factors]

    def _calibrate(U_r: torch.Tensor, S_r: torch.Tensor, V_r: torch.Tensor) -> torch.Tensor:
        return calibration_gammas(U_r, S_r, V_r, flat, alpha)

    return _calibrate
