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

    # svd_lowrank draws a random projection, so the two calls have to start from
    # the same RNG state to be comparable bit-for-bit.
    torch.manual_seed(99)
    a = merged_delta_to_lora(delta, target_rank=4, energy=None)
    torch.manual_seed(99)
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
