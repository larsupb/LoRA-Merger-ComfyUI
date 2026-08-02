# SVC — Singular Value Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Singular Value Calibration (SVC) as an opt-in post-processing step on the delta-space merge path, so that spectral directions shared by several LoRAs stop being counted multiple times in the merged delta.

**Architecture:** SVC is a pure rescaling of the singular values of the merged delta — the singular *vectors* are untouched. The repo already computes a randomized SVD of the merged delta in `merged_delta_to_lora` before refactoring it back into a LoRA, so SVC slots in as a callback at exactly that point. The per-task statistics SVC needs are computed from the LoRA factors in factored form (`U_rᵀ · up_i · down_i`), so no dense per-task delta is ever materialized and peak memory is unchanged.

**Tech Stack:** Python, PyTorch (no new dependencies), ComfyUI custom node.

**Source:** *When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging* (ICML 2026), arXiv:2602.05536. Reference implementation: `github.com/lyymuwu/SVC` (MIT).

**Branch:** cut `feat/svc-calibration` from `dev` (currently clean).

---

## Background: what SVC does and why it fits here

When several independently trained LoRAs share aligned singular directions, any linear combination of their deltas accumulates those directions repeatedly. The merged delta's singular values along shared directions are inflated, so the merged model is dominated by shared subspaces and loses task-specific behaviour. SVC measures the inflation per direction and divides it back out.

**The algorithm** (paper notation, K tasks, merged delta ΔW_m):

1. SVD the merged delta: `ΔW_m = U Σ Vᵀ`. The left singular vectors `{u^r}` are the shared output-space basis.
2. For each direction `r` and task `i`:
   - task response `a_i^r = (u^r)ᵀ ΔW_i`
   - merged response `a_m^r = (u^r)ᵀ ΔW_m`
   - scaling coefficient `s_i^r = ⟨a_m^r, a_i^r⟩ / ‖a_i^r‖²`  (>1 means over-counted)
   - aggregate: `γ^r = K / Σ_i max(α, s_i^r)`
3. Rescale: `σ̃^r = γ^r · σ^r`, reconstruct `ΔW̃_m = Σ_r σ̃^r u^r (v^r)ᵀ`.

**Three implementation facts that make this cheap in this repo:**

1. `a_m^r` never needs computing. Since `ΔW_m = U Σ Vᵀ`, we have `(u^r)ᵀ ΔW_m = σ^r (v^r)ᵀ` exactly. The merged response is a row of `diag(S) Vᵀ`, already in hand.
2. `a_i^r` never needs a dense `ΔW_i`. With `ΔW_i = scale_i · up_i · down_i`, the whole stack of responses is
   `A_i = U_rᵀ ΔW_i = scale_i · ((U_rᵀ up_i) down_i)`, an `[r, in]` matrix built from two small matmuls. Peak memory is `[r, in]` (r ≈ 32–64), not `[out, in]`.
3. The SVD is already computed. [`src/merge/lora_refactor.py:61`](../../../src/merge/lora_refactor.py#L61) has `U, S, V` in hand and slices them to rank `r`. SVC only needs to multiply `S[:r]` by a vector before line 71.

### Two deliberate deviations from the paper

Both must be preserved — the tests in this plan depend on them.

**(a) Task matrices are strength-weighted.** The paper's `ΔW_i` is `W_i − W_pre`, unweighted. In this node the user sets a per-LoRA strength, and the merged delta contains `w_i · ΔW_i`. If SVC compared the merged delta against the *unweighted* `ΔW_i`, a user strength of 0.5 would read as `s_i ≈ 0.5` — "attenuated" — and SVC would amplify it back, silently undoing the strength setting. So the task matrix for SVC is `w_i · scale_i · up_i · down_i`. With this convention `s_i = 1` means "task i appears in the merge exactly as the user asked for", which is the correct neutral point.

**(b) Rank selection runs on the uncalibrated spectrum.** `merged_delta_to_lora` picks the output rank from cumulative energy of the original `S`. Calibration can only shrink energy (γ ≤ 1 whenever α ≥ 1... and in general redistributes it), so calibrating first would move the energy target and change the output rank as a side effect. Rank is selected exactly as today, *then* the retained `S[:r]` is calibrated. SVC changes only magnitudes, never the rank or the basis.

### Behaviour this produces (these are the test cases)

| Situation | Expected γ | Why |
|---|---|---|
| Single LoRA | 1.0 | `a_1 = a_m`, so `s_1 = 1` exactly |
| Two identical LoRAs, summed (normalize=False) | 0.5 | `s_i = 2` for both → `γ = 2/(2+2)`; the duplicate is divided back out |
| Two identical LoRAs, averaged (normalize=True) | 1.0 | the average already avoids over-counting; SVC correctly does nothing |
| Two orthogonal LoRAs | 1.0 | in a direction owned by LoRA 1, LoRA 2 has zero response → neutral vote |

The orthogonal case is what fixes the zero-denominator guard: when `‖a_i^r‖² ≈ 0`, task `i` has no presence in direction `r` and must cast a **neutral vote (`s_i^r = 1`)**, not a zero one. A zero vote would clamp to `α`, inflate `γ`, and amplify directions that no task disputes. The paper does not specify this; the orthogonality test pins it down.

### The α hyperparameter

`α ∈ (0, 1]` is the floor in `max(α, s_i^r)`.

- `α = 1.0` → every term is ≥ 1, so `γ ≤ 1` always: **suppression only**, never amplification. The conservative setting.
- `α = 1/K` (paper default) → under-represented directions can push `γ` above 1, amplifying them. **Balanced.**

The node exposes these as `"suppress_only"` and `"balanced"` rather than a raw float — a bad α is hard for a user to diagnose from image output.

---

## File Structure

- **Create** `src/merge/svc.py` — all SVC math. Pure torch, no intra-package imports (matches the convention in `gta.py` and `lora_refactor.py`, so it stays loadable by file path). Responsible for: computing γ, resolving α, and building the calibrator callable.
- **Modify** `src/merge/lora_refactor.py` — add one optional `calibrate` callback parameter. Stays generic: it knows nothing about SVC, only that something may rescale the retained singular values.
- **Modify** `src/lora_mergekit_merge.py` — new `svc` widget, thread it to `process_key`, build the calibrator from the LoRA factors.
- **Create** `tests/test_svc.py` — unit tests for the math and the guards.
- **Modify** `tests/test_decomposition.py` — no. Instead **create** `tests/test_svc_refactor.py` for the `merged_delta_to_lora` integration.
- **Modify** `README.md` — document the widget.

---

## Task 1: SVC core — the γ computation

**Files:**
- Create: `src/merge/svc.py`
- Test: `tests/test_svc.py`

- [~] **Step 1: Create the branch**  *(skipped — implemented directly on `dev` at the user's request)*

```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
git checkout -b feat/svc-calibration
```

- [x] **Step 2: Write the failing test**

Create `tests/test_svc.py`:

```python
"""Unit tests for Singular Value Calibration (src/merge/svc.py)."""
import pytest
import torch

from src.merge.svc import calibration_gammas


def _svd_of(W, r):
    """Exact top-r SVD factors in the (U, S, V) convention svc expects."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :r], S[:r], Vh[:r, :].T


def test_single_task_is_identity():
    """One LoRA: the merged delta IS the task delta, so nothing is over-counted."""
    torch.manual_seed(0)
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    merged = up @ down

    U_r, S_r, V_r = _svd_of(merged, 4)
    gamma = calibration_gammas(U_r, S_r, V_r, [(up, down, 1.0)], alpha=1.0)

    assert gamma.shape == (4,)
    assert torch.allclose(gamma, torch.ones(4), atol=1e-5)


def test_duplicate_task_summed_is_halved():
    """Two identical LoRAs summed: each direction is counted twice -> gamma 0.5."""
    torch.manual_seed(1)
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    delta = up @ down
    merged = delta + delta                      # normalize=False convention

    U_r, S_r, V_r = _svd_of(merged, 4)
    factors = [(up, down, 1.0), (up, down, 1.0)]
    gamma = calibration_gammas(U_r, S_r, V_r, factors, alpha=0.5)

    assert torch.allclose(gamma, torch.full((4,), 0.5), atol=1e-5)


def test_duplicate_task_averaged_is_identity():
    """The same two LoRAs averaged are already balanced -- SVC must not touch them.

    Note the factor scales stay 1.0: normalize divides the *merge*, it is not a
    per-task strength, so it must not be folded into the task matrices. Getting
    this wrong (scale 0.5 each) would report s_i = 2 and halve an already
    correctly-balanced merge."""
    torch.manual_seed(2)
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    delta = up @ down
    merged = 0.5 * (delta + delta)              # normalize=True convention

    U_r, S_r, V_r = _svd_of(merged, 4)
    factors = [(up, down, 1.0), (up, down, 1.0)]
    gamma = calibration_gammas(U_r, S_r, V_r, factors, alpha=0.5)

    assert torch.allclose(gamma, torch.ones(4), atol=1e-5)
```

- [x] **Step 3: Run the tests to verify they fail**

```bash
PY=/home/lars/SD/Apps/ComfyUI/.venv/bin/python
$PY -m pytest tests/test_svc.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'src.merge.svc'`.

- [x] **Step 4: Write the implementation**

Create `src/merge/svc.py`:

```python
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
            actually used it (see the plan's deviation (a)).
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

        # Absent tasks vote neutrally (s = 1); see the plan's orthogonality case.
        s_i = torch.where(den > floor, num / den.clamp_min(1e-30),
                          torch.ones_like(den))
        denom_acc += s_i.clamp_min(float(alpha))

    return k / denom_acc.clamp_min(1e-8)
```

- [x] **Step 5: Run the tests to verify they pass**

```bash
$PY -m pytest tests/test_svc.py -v
```

Expected: 3 passed.

- [x] **Step 6: Commit**

```bash
git add src/merge/svc.py tests/test_svc.py
git commit -m "feat(svc): add singular value calibration core"
```

---

## Task 2: Guards — absent tasks, degenerate spectra, alpha resolution

**Files:**
- Modify: `src/merge/svc.py`
- Test: `tests/test_svc.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/test_svc.py`:

```python
from src.merge.svc import SVC_MODES, resolve_alpha


def test_orthogonal_tasks_are_untouched():
    """Block-diagonal LoRAs share no directions: absent tasks must vote neutral,
    so gamma stays 1 and neither LoRA is attenuated."""
    up1 = torch.zeros(8, 2)
    up1[0:4] = torch.randn(4, 2)
    down1 = torch.zeros(2, 8)
    down1[:, 0:4] = torch.randn(2, 4)

    up2 = torch.zeros(8, 2)
    up2[4:8] = torch.randn(4, 2)
    down2 = torch.zeros(2, 8)
    down2[:, 4:8] = torch.randn(2, 4)

    merged = up1 @ down1 + up2 @ down2
    U_r, S_r, V_r = _svd_of(merged, 4)
    factors = [(up1, down1, 1.0), (up2, down2, 1.0)]
    gamma = calibration_gammas(U_r, S_r, V_r, factors, alpha=0.5)

    assert torch.allclose(gamma, torch.ones(4), atol=1e-4)


def test_zero_task_does_not_produce_nan():
    """An all-zero LoRA has no response anywhere; it must not poison gamma."""
    torch.manual_seed(3)
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    merged = up @ down
    factors = [(up, down, 1.0), (torch.zeros(8, 4), torch.zeros(4, 6), 1.0)]

    U_r, S_r, V_r = _svd_of(merged, 4)
    gamma = calibration_gammas(U_r, S_r, V_r, factors, alpha=1.0)

    assert torch.isfinite(gamma).all()
    assert (gamma > 0).all()
    assert torch.allclose(gamma, torch.ones(4), atol=1e-5)


def test_empty_factor_list_is_identity():
    U_r, S_r, V_r = _svd_of(torch.randn(8, 6), 4)
    gamma = calibration_gammas(U_r, S_r, V_r, [], alpha=1.0)
    assert torch.allclose(gamma, torch.ones(4))


def test_suppress_only_never_amplifies():
    """alpha=1.0 floors every term at 1, so gamma can never exceed 1."""
    torch.manual_seed(4)
    up1, down1 = torch.randn(8, 4), torch.randn(4, 6)
    up2, down2 = torch.randn(8, 4), torch.randn(4, 6)
    merged = up1 @ down1 + up2 @ down2
    U_r, S_r, V_r = _svd_of(merged, 4)

    gamma = calibration_gammas(U_r, S_r, V_r,
                               [(up1, down1, 1.0), (up2, down2, 1.0)], alpha=1.0)
    assert (gamma <= 1.0 + 1e-6).all()


def test_resolve_alpha_modes():
    assert resolve_alpha("suppress_only", 4) == 1.0
    assert resolve_alpha("balanced", 4) == pytest.approx(0.25)
    assert resolve_alpha("balanced", 1) == pytest.approx(1.0)
    assert resolve_alpha("off", 4) is None


def test_resolve_alpha_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown SVC mode"):
        resolve_alpha("nonsense", 4)


def test_svc_modes_tuple_is_the_widget_contract():
    assert SVC_MODES == ("off", "balanced", "suppress_only")
```

- [x] **Step 2: Run to verify they fail**

```bash
$PY -m pytest tests/test_svc.py -v
```

Expected: `ImportError: cannot import name 'SVC_MODES'` (collection error for the whole file).

- [x] **Step 3: Add the mode helpers**

Append to `src/merge/svc.py`:

```python
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
```

- [x] **Step 4: Run to verify they pass**

```bash
$PY -m pytest tests/test_svc.py -v
```

Expected: 10 passed. The guard tests pass against the Task 1 implementation — `_ABSENT_REL` and the `clamp_min` were written for them.

- [x] **Step 5: Commit**

```bash
git add src/merge/svc.py tests/test_svc.py
git commit -m "feat(svc): add mode/alpha resolution and guard coverage"
```

---

## Task 3: The calibrator factory

**Files:**
- Modify: `src/merge/svc.py`
- Test: `tests/test_svc.py`

- [x] **Step 1: Write the failing test**

Append to `tests/test_svc.py`:

```python
from src.merge.svc import make_calibrator


def test_make_calibrator_returns_none_when_off():
    assert make_calibrator([(torch.randn(8, 4), torch.randn(4, 6), 1.0)], "off") is None


def test_make_calibrator_matches_direct_call():
    torch.manual_seed(5)
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    merged = 2 * (up @ down)
    factors = [(up, down, 1.0), (up, down, 1.0)]

    U_r, S_r, V_r = _svd_of(merged, 4)
    calibrate = make_calibrator(factors, "balanced")
    direct = calibration_gammas(U_r, S_r, V_r, factors, alpha=0.5)

    assert torch.allclose(calibrate(U_r, S_r, V_r), direct)


def test_make_calibrator_handles_4d_conv_factors():
    """Conv LoRAs arrive as 4D; the calibrator flattens them the same way the
    merged delta is flattened in merged_delta_to_lora."""
    torch.manual_seed(6)
    up4 = torch.randn(8, 4, 1, 1)
    down4 = torch.randn(4, 3, 3, 3)
    up2 = up4.reshape(8, -1)
    down2 = down4.reshape(4, -1)
    merged = up2 @ down2

    U_r, S_r, V_r = _svd_of(merged, 4)
    calibrate = make_calibrator([(up4, down4, 1.0)], "suppress_only")

    assert torch.allclose(calibrate(U_r, S_r, V_r), torch.ones(4), atol=1e-5)
```

- [x] **Step 2: Run to verify they fail**

```bash
$PY -m pytest tests/test_svc.py -v
```

Expected: `ImportError: cannot import name 'make_calibrator'`.

- [x] **Step 3: Implement the factory**

Append to `src/merge/svc.py`:

```python
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
```

- [x] **Step 4: Run to verify they pass**

```bash
$PY -m pytest tests/test_svc.py -v
```

Expected: 13 passed.

- [x] **Step 5: Commit**

```bash
git add src/merge/svc.py tests/test_svc.py
git commit -m "feat(svc): add calibrator factory with conv flattening"
```

---

## Task 4: Wire calibration into the refactor step

**Files:**
- Modify: `src/merge/lora_refactor.py:18-22` (signature) and `:71` (spectrum use)
- Test: `tests/test_svc_refactor.py`

- [x] **Step 1: Write the failing test**

Create `tests/test_svc_refactor.py`:

```python
"""Integration: SVC calibration applied inside merged_delta_to_lora."""
import torch

from src.merge.lora_refactor import merged_delta_to_lora
from src.merge.svc import make_calibrator


def _delta_of(up, down, alpha):
    rank = up.shape[1]
    return (float(alpha) / rank) * (up @ down)


def test_calibrate_none_is_unchanged_behaviour():
    torch.manual_seed(10)
    delta = torch.randn(16, 4) @ torch.randn(4, 12)

    a = merged_delta_to_lora(delta, target_rank=4, energy=None)
    b = merged_delta_to_lora(delta, target_rank=4, energy=None, calibrate=None)

    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])
    assert a[2] == b[2]


def test_calibration_undoes_a_duplicated_lora():
    """Summing a LoRA with itself doubles it. SVC should pull the merged delta
    back to (approximately) one copy."""
    torch.manual_seed(11)
    up = torch.randn(16, 4)
    down = torch.randn(4, 12)
    single = up @ down
    merged = single + single

    calibrate = make_calibrator([(up, down, 1.0), (up, down, 1.0)], "balanced")
    out_up, out_down, alpha_out = merged_delta_to_lora(
        merged, target_rank=4, energy=None, calibrate=calibrate)

    recon = _delta_of(out_up, out_down, alpha_out)
    rel = (recon - single).norm() / single.norm()
    assert rel < 1e-3, f"relative error {rel:.4f}"


def test_calibration_does_not_change_the_output_rank():
    """Rank is chosen from the uncalibrated energy spectrum, so enabling SVC must
    not change how many components come out."""
    torch.manual_seed(12)
    up = torch.randn(16, 6)
    down = torch.randn(6, 12)
    merged = 2 * (up @ down)
    factors = [(up, down, 1.0), (up, down, 1.0)]

    plain = merged_delta_to_lora(merged, target_rank=6, energy=0.99)
    calibrated = merged_delta_to_lora(
        merged, target_rank=6, energy=0.99,
        calibrate=make_calibrator(factors, "balanced"))

    assert plain[0].shape == calibrated[0].shape
    assert plain[2] == calibrated[2]


def test_zero_delta_shortcircuits_before_calibration():
    """The all-zero early return happens before the SVD, so a calibrator that
    would explode if called must never be called."""
    def boom(*args):
        raise AssertionError("calibrate must not run on the zero-delta path")

    up, down, alpha_out = merged_delta_to_lora(
        torch.zeros(8, 6), target_rank=4, calibrate=boom)

    assert up.shape == (8, 1)
    assert down.shape == (1, 6)
    assert alpha_out == 1.0
```

- [x] **Step 2: Run to verify it fails**

```bash
$PY -m pytest tests/test_svc_refactor.py -v
```

Expected: `TypeError: merged_delta_to_lora() got an unexpected keyword argument 'calibrate'`.

- [x] **Step 3: Add the parameter**

In `src/merge/lora_refactor.py`, replace the import line and the signature (lines 13-22):

```python
from typing import Callable, Optional, Tuple

import torch

Calibrator = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def merged_delta_to_lora(delta: torch.Tensor, target_rank: int, *,
                         energy: Optional[float] = 0.99, device="cpu",
                         dtype: torch.dtype = torch.float32,
                         oversample: int = 8, niter: int = 2,
                         calibrate: Optional[Calibrator] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, float]:
```

Add to the docstring's Args section, after the `niter` entry:

```
        calibrate: optional ``(U_r, S_r, V_r) -> gamma`` hook that rescales the
            retained singular values (see src/merge/svc.py). Applied *after* the
            rank is chosen, so it changes magnitudes only -- never the rank or
            the singular vectors. ``None`` reproduces the uncalibrated path
            exactly.
```

- [x] **Step 4: Apply the calibration**

In `src/merge/lora_refactor.py`, replace line 71:

```python
    s_sqrt = S[:r].clamp_min(0).sqrt()
```

with:

```python
    S_r = S[:r]
    if calibrate is not None:
        gamma = calibrate(U[:, :r], S_r, V[:, :r])
        S_r = S_r * gamma.to(device=S_r.device, dtype=S_r.dtype)
    s_sqrt = S_r.clamp_min(0).sqrt()
```

- [x] **Step 5: Run to verify they pass**

```bash
$PY -m pytest tests/test_svc_refactor.py -v
```

Expected: 4 passed.

- [x] **Step 6: Run the full suite for regressions**

```bash
$PY -m pytest tests/
```

Expected: 255 passed (238 before + 13 in `test_svc.py` + 4 in `test_svc_refactor.py`). No failures.

- [x] **Step 7: Commit**

```bash
git add src/merge/lora_refactor.py tests/test_svc_refactor.py
git commit -m "feat(svc): accept a spectrum calibrator in merged_delta_to_lora"
```

---

## Task 5: Expose SVC on the merger node

**Files:**
- Modify: `src/lora_mergekit_merge.py:16-17` (imports), `:89-113` (INPUT_TYPES), `:136-197` (`lora_mergekit`), `:199-200` (`merge`), `:347-349` (call site)
- Test: `tests/test_svc_node.py`

- [x] **Step 1: Write the failing test**

Create `tests/test_svc_node.py`:

```python
"""The merger node's SVC widget and its factor plumbing."""
import torch

from src.lora_mergekit_merge import LoraMergekitMerge, build_svc_factors
from src.merge.svc import SVC_MODES


def test_widget_offers_the_svc_modes():
    spec = LoraMergekitMerge.INPUT_TYPES()["required"]
    assert "svc" in spec
    choices, opts = spec["svc"]
    assert tuple(choices) == SVC_MODES
    assert opts["default"] == "off"


def test_build_svc_factors_folds_alpha_and_strength_into_scale():
    """scale must be (alpha/rank) * strength so the factor reproduces the task's
    contribution exactly as the merge used it."""
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    tuples = {"a": (up, down, torch.tensor(2.0))}
    weights = torch.tensor([3.0])

    factors = build_svc_factors(tuples, weights)

    assert len(factors) == 1
    f_up, f_down, scale = factors[0]
    assert f_up is up and f_down is down
    assert scale == 2.0 / 4 * 3.0


def test_build_svc_factors_defaults_alpha_to_rank():
    """alpha=None means scale 1.0 -- the same convention as the delta builder."""
    up = torch.randn(8, 4)
    down = torch.randn(4, 6)
    factors = build_svc_factors({"a": (up, down, None)}, torch.tensor([1.0]))
    assert factors[0][2] == 1.0


def test_build_svc_factors_preserves_order():
    ups = [torch.randn(8, 2) for _ in range(3)]
    downs = [torch.randn(2, 6) for _ in range(3)]
    tuples = {f"l{i}": (ups[i], downs[i], None) for i in range(3)}
    factors = build_svc_factors(tuples, torch.tensor([1.0, 2.0, 3.0]))

    assert [f[2] for f in factors] == [1.0, 2.0, 3.0]
    assert all(factors[i][0] is ups[i] for i in range(3))
```

- [x] **Step 2: Run to verify it fails**

```bash
$PY -m pytest tests/test_svc_node.py -v
```

Expected: `ImportError: cannot import name 'build_svc_factors'`.

- [x] **Step 3: Add the factor builder**

In `src/lora_mergekit_merge.py`, extend the existing import at line 17 and add the helper right after the imports block (before the node class):

```python
from .merge.lora_refactor import merged_delta_to_lora
from .merge.svc import SVC_MODES, make_calibrator
```

```python
def build_svc_factors(lora_key_tuples, weights):
    """Per-task ``(up, down, scale)`` factors for SVC, in stack order.

    ``scale`` folds in both the LoRA's own ``alpha/rank`` and the user's merge
    strength, so ``scale * up @ down`` is exactly the contribution the merge saw
    (mirrors the delta construction in ``process_key``). SVC compares against the
    strength-weighted task delta on purpose: measured against the unweighted one,
    a user strength below 1 would read as attenuation and get amplified back.
    """
    factors = []
    for i, (u, d, a) in enumerate(lora_key_tuples.values()):
        rank = u.shape[1]
        scale = (float(a) / rank) if a is not None else 1.0
        factors.append((u, d, scale * float(weights[i])))
    return factors
```

- [x] **Step 4: Add the widget**

In `src/lora_mergekit_merge.py`, insert into `INPUT_TYPES`'s `"required"` dict immediately after the `refactor_method` entry (after line 103):

```python
                "svc": (
                    list(SVC_MODES),
                    {
                        "default": "off",
                        "tooltip": (
                            "Singular Value Calibration on the merged delta "
                            "(delta-space methods only: the GTA family, SCE and "
                            "the interpolation methods).\n"
                            "LoRAs that share singular directions have those "
                            "directions counted once per LoRA, so the merge is "
                            "dominated by what they have in common. SVC measures "
                            "the over-counting per direction and rescales the "
                            "singular values to undo it. Singular vectors and "
                            "output rank are unchanged.\n"
                            "'off': disabled.\n"
                            "'balanced': paper default (alpha = 1/K); can both "
                            "suppress over-counted and amplify under-represented "
                            "directions.\n"
                            "'suppress_only': alpha = 1; only ever scales down. "
                            "Start here if 'balanced' over-sharpens."
                        ),
                    },
                ),
```

- [x] **Step 5: Thread the parameter through**

In `src/lora_mergekit_merge.py`:

Add `svc: str = "off"` to the `lora_mergekit` signature after `refactor_method` (line 144):

```python
                      refactor_method: str = "energy_rSVD",
                      svc: str = "off",
                      offload_models: bool = True):
```

Pass it to `self.merge` (line 180-182):

```python
        merge = self.merge(method=merge_method, method_args=method_args, lambda_=output_scale,
                           spectral_norm_scale=spectral_norm_scale, merge_clip=merge_clip,
                           device=device, dtype=dtype, refactor_method=refactor_method,
                           svc=svc)
```

Record it in `merge_context` (after line 194):

```python
            "refactor_method": refactor_method,
            "svc": svc,
```

Extend the `merge` signature (line 199-200):

```python
    def merge(self, method: MergeMethod, method_args, lambda_, spectral_norm_scale, merge_clip, device, dtype,
              refactor_method: str = "energy_rSVD", svc: str = "off"):
```

- [x] **Step 6: Use it at the call site**

In `src/lora_mergekit_merge.py`, replace lines 347-349:

```python
                target_rank = max(u.shape[1] for (u, _, _) in lora_key_tuples.values())
                up, down, alpha_out = merged_delta_to_lora(
                    merged, target_rank=target_rank, energy=refactor_energy, device=device)
```

with:

```python
                target_rank = max(u.shape[1] for (u, _, _) in lora_key_tuples.values())
                # The factors are still live here -- gta_merge consumed `deltas`,
                # not lora_key_tuples -- so SVC's per-task responses are built in
                # factored form and no dense per-task delta is resurrected.
                calibrate = make_calibrator(
                    build_svc_factors(lora_key_tuples, weights), svc)
                up, down, alpha_out = merged_delta_to_lora(
                    merged, target_rank=target_rank, energy=refactor_energy,
                    device=device, calibrate=calibrate)
```

- [x] **Step 7: Run the node tests**

```bash
$PY -m pytest tests/test_svc_node.py -v
```

Expected: 4 passed.

- [x] **Step 8: Run the full suite**

```bash
$PY -m pytest tests/
```

Expected: 259 passed. If `tests/test_merger_vram_offload.py` or `tests/test_merge_node_names.py` fail on the new widget, they assert on `INPUT_TYPES` contents — update those assertions to include `svc`, do not remove the widget.

- [x] **Step 9: Commit**

```bash
git add src/lora_mergekit_merge.py tests/test_svc_node.py
git commit -m "feat(svc): expose singular value calibration on the merger node"
```

---

## Task 6: Documentation

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add the feature bullet**

In `README.md`, under `## Features`, insert after the "Spectral Norm Regularization" bullet:

```markdown
- **Singular Value Calibration (SVC)**: Opt-in post-merge calibration that detects singular directions shared by several LoRAs and rescales the merged spectrum so they stop being counted once per LoRA. Training-free, data-free, no rank or basis change ([arXiv:2602.05536](https://arxiv.org/abs/2602.05536))
```

- [x] **Step 2: Document the widget**

In `README.md`, in the **PM LoRA Merger** node section, add after the `refactor_method` parameter description:

```markdown
- `svc`: Singular Value Calibration on the merged delta. Applies to the delta-space methods (the GTA family, SCE, and the interpolation methods); CLIP layers are unaffected.
  - `off` (default): no calibration.
  - `balanced`: paper default (α = 1/K). Suppresses over-counted directions and can amplify under-represented ones.
  - `suppress_only`: α = 1, so the calibration can only scale directions *down*. Use this if `balanced` over-sharpens.

  Why it exists: independently trained LoRAs tend to reuse a small set of singular directions. A linear merge accumulates those directions once per LoRA, inflating their singular values, and the merged LoRA drifts toward whatever its inputs have in common at the expense of what makes each one distinct. SVC measures the inflation per direction and divides it back out. Singular vectors and output rank are untouched — only the magnitudes change. Merging a LoRA with a copy of itself is the clearest case: with `svc: off` the result is 2x the original, with SVC it comes back to 1x.
```

- [x] **Step 3: Verify nothing else regressed**

```bash
$PY -m pytest tests/
```

Expected: 259 passed.

- [x] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document the SVC widget on the merger node"
```

---

## Task 7: Manual verification in ComfyUI

Automated tests prove the math. This task proves the node works in the real app. It is not optional — nothing before this has run a single LoRA through ComfyUI.

- [ ] **Step 1: Restart ComfyUI and load a two-LoRA merge workflow**

Use `assets/pm-basic_workflow.png` as the starting workflow. Set the merge method to **PM TIES**, `device: cuda`, `refactor_method: energy_rSVD`.

- [ ] **Step 2: Confirm the widget appears**

The **PM LoRA Merger** node must show an `svc` dropdown with `off` / `balanced` / `suppress_only`, defaulting to `off`.

- [ ] **Step 3: Baseline**

Run with `svc: off`. Save the output image and note the wall-clock merge time from the ComfyUI console.

- [ ] **Step 4: Duplicate-LoRA sanity check**

Load the *same* LoRA twice into the stacker, both at strength 1.0, with `normalize` disabled on the method node. Merge with `svc: off`, then with `svc: balanced`, at a fixed seed.

Expected: the `off` result is visibly over-baked (the LoRA applied at roughly double strength); the `balanced` result closely resembles applying that LoRA once. This is the property `test_calibration_undoes_a_duplicated_lora` asserts numerically.

- [ ] **Step 5: Real two-LoRA merge**

With two *different* LoRAs, generate at a fixed seed for all three `svc` settings. Expected: `balanced` and `suppress_only` retain more of each LoRA's distinct character than `off`. Overhead should be a few percent of merge time at most — if it is not, stop and profile `calibration_gammas`.

- [ ] **Step 6: Record the result**

Append an "SVC" section to this plan file with: the three images (or a description of the differences), the merge times, and whether `balanced` or `suppress_only` looked better. If SVC makes output *worse* on real LoRA pairs, say so plainly and leave the default at `off` — the paper's gains are reported on ViT and LLM benchmarks, not on diffusion models, and that transfer is exactly what this step tests.

- [ ] **Step 7: Commit the findings**

```bash
git add docs/superpowers/plans/2026-08-02-svc-singular-value-calibration.md
git commit -m "docs: record SVC manual verification results"
```

---

## SVC — verification results (2026-08-02)

**Status: Tasks 1–6 complete and committed on `dev` (no feature branch, per request).
Task 7 is NOT done — steps 1–6 need a running ComfyUI with a GPU and a human
looking at images, which the implementation session could not do.**

What *was* verified, and how:

### Automated (`$PY -m pytest tests/`)

259 passed, 0 failed (238 before this work + 13 in `test_svc.py` + 4 in
`test_svc_refactor.py` + 4 in `test_svc_node.py`). One pre-existing assertion was
updated rather than the widget removed: `tests/test_merge_node_names.py::test_merger_output_scale`
pins the exact merger widget order, so `"svc"` was inserted between
`refactor_method` and `offload_models`.

One test deviates from the plan's text. `test_calibrate_none_is_unchanged_behaviour`
compares two `merged_delta_to_lora` calls bit-for-bit, but `torch.svd_lowrank` draws
a random projection, so back-to-back calls differ (~1e-3 relative) with or without
SVC. The test now seeds the RNG identically before each call; without that it fails
for reasons unrelated to calibration.

The plan's test file also names the node class `LoraMergekitMerge`; the actual class
is `LoraMergerMergekit`, and `tests/test_svc_node.py` uses the real name.

### End-to-end through `LoraMergerMergekit.merge()` (headless, CPU, TIES,
`average_weights` off, rank-8 synthetic LoRAs)

This is a numeric stand-in for Task 7 steps 4–5 — it exercises the real node path
(widget value → `build_svc_factors` → `make_calibrator` → `merged_delta_to_lora`),
just without a diffusion model or images.

| Case | `svc` | ‖merged‖ / ‖single‖ | rel. change vs `off` |
|---|---|---|---|
| Same LoRA twice | `off` | 2.0000 | — |
| Same LoRA twice | `balanced` | 1.0000 | rel. err vs one copy: 0.00000 |
| Same LoRA twice | `suppress_only` | 1.0000 | rel. err vs one copy: 0.00000 |
| Two different LoRAs | `balanced` | — | 0.158 |
| Two different LoRAs | `suppress_only` | — | 0.158 |
| Single LoRA | `balanced` / `suppress_only` | — | 0.000001 (no-op) |

The duplicate case behaves exactly as designed: `off` gives 2× the original delta,
both SVC modes bring it back to precisely 1×. A single LoRA is untouched.

`balanced` and `suppress_only` coincide on two independent random LoRAs: every
`s_i` came out ≥ 1, so the α floor never binds and the two modes compute the same
γ. They will diverge only when some task is genuinely under-represented in a
direction. The 0.158 figure is well clear of the ~0.0014 run-to-run noise floor of
the randomized SVD (measured by comparing two `off` runs), so the calibration is
doing real work, not drifting.

Overhead is negligible: 1–3 ms per layer either way on these shapes, no measurable
difference between `off` and the SVC modes.

### Still open (needs a human at ComfyUI)

- Widget renders correctly in the **PM LoRA Merger** node UI.
- Whether SVC *helps image quality* on real diffusion LoRA pairs, and whether
  `balanced` or `suppress_only` looks better. The paper's gains are on ViT/LLM
  benchmarks; the transfer to diffusion is untested here. The default stays `off`
  until someone runs step 5.
- Overhead on real FLUX-sized layers under CUDA (the numbers above are small CPU
  matrices).

---

## Out of scope

- **Iso-C / Iso-CTS.** Uniform spectrum flattening is a sibling of SVC at the same insertion point. If SVC's `calibrate` hook lands, Iso-C is a second calibrator returning `gamma = target / S_r`. Not built here.
- **Applying SVC to the factored (non-delta-space) path.** Methods that merge `up`/`down` separately never form a merged delta, so there is no spectrum to calibrate.
- **CLIP layers.** They use a weighted average and never reach `merged_delta_to_lora`.
- **Per-layer α.** One global mode for now. Revisit only if Task 7 shows the calibration helping some layer groups and hurting others.
