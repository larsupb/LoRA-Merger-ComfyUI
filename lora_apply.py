import logging

from .utility import load_as_comfy_lora


class LoraApply:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP",),
                             "lora": ("LoRA",),
                             }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_merged_lora"
    CATEGORY = "LoRA PowerMerge"

    def apply_merged_lora(self, model, clip, lora):
        strength_model = lora["strength_model"]
        strength_clip = lora["strength_clip"]

        if strength_model == 0 and strength_clip == 0:
            return model, clip

        if 'lora' not in lora or lora['lora'] is None:
            lora['lora'] = load_as_comfy_lora(lora, model, clip)

        new_model_patcher = model.clone()
        k = new_model_patcher.add_patches(lora['lora'], strength_model)
        new_clip = clip.clone()
        k1 = new_clip.add_patches(lora['lora'], strength_clip)

        k = set(k)
        k1 = set(k1)
        for x in lora["lora"]:
            if (x not in k) and (x not in k1):
                logging.warning("PM LoraApply: NOT LOADED {}".format(x))

        return new_model_patcher, new_clip
