import logging

from .comfy_util import load_as_comfy_lora


class LoraApply:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "lora": ("LoRABundle",),
                             }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_merged_lora"
    CATEGORY = "LoRA PowerMerge"

    def apply_merged_lora(self, model, lora):
        strength_model = lora["strength_model"]

        if strength_model == 0:
            return (model,)

        if 'lora' not in lora or lora['lora'] is None:
            lora['lora'] = load_as_comfy_lora(lora, model)

        new_model_patcher = model.clone()
        k = new_model_patcher.add_patches(lora['lora'], strength_model)

        k = set(k)
        for x in lora["lora"]:
            if x not in k:
                logging.warning("PM LoraApply: NOT LOADED {}".format(x))

        return (new_model_patcher,)
