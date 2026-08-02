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
