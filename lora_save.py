import os

import comfy
import folder_paths


from .sd_lora import convert_to_regular_lora

class LoraSave:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora": ("LoRA",),
            "file_name": ("STRING", {"multiline": False, "default": "merged"}), "extension": (["safetensors"], ),
        }}
    RETURN_TYPES = ()
    FUNCTION = "lora_save"
    CATEGORY = "LoRA PowerMerge"

    OUTPUT_NODE = True

    def lora_save(self, model, clip, lora, file_name, extension):
        save_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0], file_name + "." + extension)

        state_dict = lora['lora']
        new_state_dict = convert_to_regular_lora(model, clip, state_dict)

        print(f"Saving LoRA to {save_path}")
        comfy.utils.save_torch_file(new_state_dict, save_path)

        return {}
