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
