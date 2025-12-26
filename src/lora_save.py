import os

import comfy
import folder_paths


from .architectures.sd_lora import convert_to_regular_lora

class LoraSave:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "lora": ("LoRABundle",),
            "file_name": ("STRING", {"multiline": False, "default": "merged"}), "extension": (["safetensors"], ),
        }}
    RETURN_TYPES = ()
    FUNCTION = "lora_save"
    CATEGORY = "LoRA PowerMerge"

    OUTPUT_NODE = True

    def lora_save(self, model, lora, file_name, extension):
        save_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0], file_name + "." + extension)

        # Convert model weights from ComfyUI format to regular LoRA format
        state_dict = lora['lora']
        new_state_dict = convert_to_regular_lora(model, state_dict)

        # CLIP layers are now merged as part of the merge process,
        # so we don't need to copy them from lora_raw anymore.
        # The merged CLIP weights are already in state_dict.

        print(f"Saving LoRA to {save_path}")
        comfy.utils.save_torch_file(new_state_dict, save_path)

        return {}
