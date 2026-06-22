import copy

import comfy
import comfy.model_management


def rebuild_guider_with_patches(guider, new_model_patcher):
    """
    Rebuild a guider around a LoRA-patched (positive) model while preserving the
    guider's exact subclass and state.

    The samplers in this package apply LoRA patches by cloning the guider's
    model_patcher and adding patches, then need a guider that wraps the patched
    model. Constructing a plain ``CFGGuider`` would silently discard any
    specialized guider behavior:

    - ``Guider_DualModel`` (DualModelGuider node) runs the negative/uncond pass
      on a *separate* ``uncond_model_patcher``. A plain CFGGuider would instead
      run uncond on the patched positive model, changing the result.
    - ``Guider_DualCFG`` (DualCFGGuider node) carries ``cfg1``/``cfg2``/``nested``.

    By shallow-copying the original guider we keep its class and every
    instance attribute (including ``cfg`` and any separate uncond model), then we
    only swap in the patched positive model and reset the conds to a clean copy.
    The LoRA patch is applied to the positive model only; a separate negative/
    uncond model, if any, is intentionally left untouched.

    Args:
        guider: The original guider passed into the sampler.
        new_model_patcher: The cloned, LoRA-patched ModelPatcher for the positive model.

    Returns:
        A new guider of the same type, wrapping ``new_model_patcher``.
    """
    new_guider = copy.copy(guider)
    new_guider.model_patcher = new_model_patcher
    new_guider.model_options = new_model_patcher.model_options
    # Fresh cond dicts (copied lists) so per-iteration sampling never mutates the
    # original guider's conds.
    new_guider.original_conds = {k: list(v) for k, v in guider.original_conds.items()}
    new_guider.conds = {k: list(v) for k, v in guider.original_conds.items()}
    return new_guider


def load_as_comfy_lora(lora: dict, model, clip=None):
    """
    Load LoRA from raw state dict into ComfyUI patch format.

    Extracts both UNet and CLIP layers from the raw LoRA state dict using
    key mappings from the provided model and clip objects.

    Args:
        lora: LoRA dictionary containing 'lora_raw' key with raw state dict
        model: MODEL to extract UNet key mappings from
        clip: CLIP model to extract CLIP key mappings from (optional but recommended)

    Returns:
        Patch dictionary with UNet layers (and CLIP layers if clip provided)

    Note:
        Without CLIP, only UNet layers will be loaded and CLIP layers will be
        skipped with "lora key not loaded" warnings. Always provide clip when
        working with LoRAs that contain CLIP weights.
    """
    if 'lora_raw' not in lora or lora['lora_raw'] is None:
        raise ValueError("LoRA data is missing. Please provide a valid LoRA dictionary with 'lora_raw' key.")

    # Build key_map for both UNet and CLIP
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    patch_dict = comfy.lora.load_lora(lora['lora_raw'], key_map)
    return patch_dict
