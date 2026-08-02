# tests/test_merge_node_names.py
# Standalone script test (repo pytest is broken). Loads the custom-node package
# under a synthetic name so the relative imports resolve, then asserts the
# renamed widget keys and the unchanged internal settings/context keys.
import importlib.util, os, sys, inspect, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)            # .../custom_nodes
COMFY_ROOT = os.path.dirname(PARENT)      # ComfyUI root, so `import comfy` resolves
PKG = "LoRA_Merger_ComfyUI_test"

sys.path.insert(0, COMFY_ROOT)            # lora_mergekit_merge imports comfy.*
sys.path.insert(0, PARENT)
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_test.src import nodes_merge_methods as N
from LoRA_Merger_ComfyUI_test.src.lora_mergekit_merge import LoraMergerMergekit

EXPECTED_ORDER = {
    "TaskArithmeticMergeMethod": ["rescale_norm", "average_weights"],
    "TIESMergeMethod": ["rescale_norm", "average_weights", "density"],
    "DAREMergeMethod": ["sign_consensus", "rescale_norm", "density", "average_weights"],
    "BreadcrumbsMergeMethod": ["sign_consensus", "rescale_norm", "density", "gamma", "average_weights"],
    "DELLAMergeMethod": ["sign_consensus", "rescale_norm", "density", "epsilon", "average_weights"],
}


def test_input_types_renamed_same_order():
    for cls_name, expected in EXPECTED_ORDER.items():
        cls = getattr(N, cls_name)
        keys = list(cls.INPUT_TYPES()["required"].keys())
        assert keys == expected, f"{cls_name}: {keys} != {expected}"


def test_settings_internal_keys_unchanged():
    # average_weights -> internal "normalize"; sign_consensus -> "sign_consensus_algorithm"
    for cls_name in EXPECTED_ORDER:
        cls = getattr(N, cls_name)
        s = cls().get_method(average_weights=False)[0]["settings"]
        assert s["normalize"] is False, f"{cls_name}: normalize not wired from average_weights"
    for cls_name in ("DAREMergeMethod", "BreadcrumbsMergeMethod", "DELLAMergeMethod"):
        cls = getattr(N, cls_name)
        s = cls().get_method(sign_consensus=True)[0]["settings"]
        assert s["sign_consensus_algorithm"] is True, f"{cls_name}: sign_consensus not wired"


def test_average_weights_defaults_off():
    # Default OFF matches ComfyUI's native additive LoRA stacking (sum, not average).
    for cls_name in EXPECTED_ORDER:
        cls = getattr(N, cls_name)
        default = cls.INPUT_TYPES()["required"]["average_weights"][1]["default"]
        assert default is False, f"{cls_name}: average_weights default {default!r} != False"


def test_sce_additive_default_and_wiring():
    # SCE exposes select_topk + average_weights; average_weights wires to the
    # internal "normalize" and defaults OFF (additive), like the GTA family.
    cls = N.SCEMergeMethod
    keys = list(cls.INPUT_TYPES()["required"].keys())
    assert keys == ["select_topk", "average_weights"], keys
    assert cls.INPUT_TYPES()["required"]["average_weights"][1]["default"] is False
    s = cls().get_method(select_topk=0.5, average_weights=False)[0]["settings"]
    assert s["normalize"] is False, f"SCE: normalize not wired from average_weights ({s})"
    assert s["select_topk"] == 0.5, s
    # ON -> normalized average
    s_on = cls().get_method(select_topk=1.0, average_weights=True)[0]["settings"]
    assert s_on["normalize"] is True, s_on


def test_interp_nodes_average_weights_wiring():
    cases = {
        "SLERPMergeMethod": dict(),
        "NuSlerpMergeMethod": dict(),
        "KArcherMergeMethod": dict(),
        "NearSwapMergeMethod": dict(),
    }
    for cls_name in cases:
        cls = getattr(N, cls_name)
        req = cls.INPUT_TYPES()["required"]
        assert "average_weights" in req, f"{cls_name}: no average_weights widget"
        assert req["average_weights"][1]["default"] is False, \
            f"{cls_name}: average_weights default != False"
        s_off = cls().get_method(average_weights=False)[0]["settings"]
        assert s_off["normalize"] is False, f"{cls_name}: normalize not wired ({s_off})"
        s_on = cls().get_method(average_weights=True)[0]["settings"]
        assert s_on["normalize"] is True, f"{cls_name}: normalize ON not wired"


def test_merger_output_scale():
    keys = list(LoraMergerMergekit.INPUT_TYPES()["required"].keys())
    expected = ["method", "components", "strengths", "output_scale",
                "spectral_norm_scale", "merge_clip", "device", "dtype", "refactor_method",
                "svc", "offload_models"]
    assert keys == expected, f"merger widget order {keys} != {expected}"
    params = inspect.signature(LoraMergerMergekit.lora_mergekit).parameters
    assert "output_scale" in params and "lambda_" not in params, list(params)


def run():
    failed = 0
    for name, fn in [
        ("input_types_renamed_same_order", test_input_types_renamed_same_order),
        ("settings_internal_keys_unchanged", test_settings_internal_keys_unchanged),
        ("average_weights_defaults_off", test_average_weights_defaults_off),
        ("sce_additive_default_and_wiring", test_sce_additive_default_and_wiring),
        ("merger_output_scale", test_merger_output_scale),
        ("interp_nodes_average_weights_wiring", test_interp_nodes_average_weights_wiring),
    ]:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED"); sys.exit(1)
    print(f"\nAll 6 passed")


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run()
