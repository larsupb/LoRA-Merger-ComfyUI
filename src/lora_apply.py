import logging
import comfy.sd

from .comfy_util import load_as_comfy_lora


class LoraApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the merged LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the merged CLIP layers will be applied to."}),
                "lora": ("LoRABundle", {"tooltip": "The merged LoRA bundle from the merger node."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "apply_merged_lora"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = """Apply merged LoRA to both MODEL and CLIP.

Applies the merged LoRA patches to both the diffusion model (UNet) and CLIP text encoder. The merged LoRA contains both UNet and CLIP weights that were merged using the selected algorithm.

Inputs:
- model: Diffusion model to apply UNet LoRA patches to
- clip: CLIP model to apply CLIP LoRA patches to
- lora: Merged LoRA bundle from PM LoRA Merger

Outputs:
- MODEL: Modified diffusion model with LoRA applied
- CLIP: Modified CLIP model with LoRA applied

This matches ComfyUI's standard LoraLoader behavior with both MODEL and CLIP inputs/outputs."""

    def apply_merged_lora(self, model, clip, lora):
        print("I AM APPLYING HERE!!!!!!!!!!!")

        strength_model = lora.get("strength_model", 1.0)
        strength_clip = lora.get("strength_clip", 1.0)

        # If both strengths are zero, return unchanged models
        if strength_model == 0 and strength_clip == 0:
            return model, clip

        # Get the LoRA patches - either from merger or from selector
        # The patches are already in ComfyUI LoRAAdapter format
        if 'lora' not in lora or lora['lora'] is None:
            # Fallback: if lora_raw exists, load it (for backward compatibility)
            if 'lora_raw' in lora and lora['lora_raw'] is not None:
                logging.info("PM LoraApply: Loading from lora_raw (fallback path)")
                lora['lora'] = load_as_comfy_lora(lora, model, clip)
            else:
                raise ValueError("LoRA bundle does not contain patches ('lora' key) or raw data ('lora_raw' key)")

        patch_dict = lora['lora']

        # Clone models
        new_model = model.clone()
        new_clip = clip.clone()

        # Apply patches to model (UNet layers)
        # The patches are already in the correct format (key -> LoRAAdapter)
        k = new_model.add_patches(patch_dict, strength_model)

        # Log any keys that weren't loaded
        loaded_keys = set(k)
        for key in patch_dict.keys():
            if key not in loaded_keys:
                logging.debug(f"PM LoraApply: Model patch not loaded: {key}")

        # Apply patches to CLIP
        # CLIP uses the same patch dict but with strength_clip
        k_clip = new_clip.add_patches(patch_dict, strength_clip)

        # Log CLIP keys that weren't loaded
        loaded_clip_keys = set(k_clip)
        for key in patch_dict.keys():
            if key not in loaded_clip_keys and key not in loaded_keys:
                # Only log if not loaded by either model or clip
                logging.debug(f"PM LoraApply: CLIP patch not loaded: {key}")

        return new_model, new_clip
