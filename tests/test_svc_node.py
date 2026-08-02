"""The merger node's SVC widget and its factor plumbing."""
import torch

from src.lora_mergekit_merge import LoraMergerMergekit, build_svc_factors
from src.merge.svc import SVC_MODES


def test_widget_offers_the_svc_modes():
    spec = LoraMergerMergekit.INPUT_TYPES()["required"]
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
