import os

import comfy
import folder_paths


class LoraPowerMergeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LoRABundle",)
    FUNCTION = "load_lora"
    CATEGORY = "LoRA PowerMerge"

    def load_lora(self, lora_name, strength_model):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_name_pretty = os.path.splitext(lora_name)[0]

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        return ({"lora_raw": lora,
                 "strength_model": strength_model,
                 "name": lora_name_pretty},)
