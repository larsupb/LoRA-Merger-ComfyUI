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

        # If lora_raw exists, extract and merge CLIP weights
        if 'lora_raw' in lora and lora['lora_raw'] is not None:
            lora_raw = lora['lora_raw']

            # Extract CLIP weights from original LoRA
            # CLIP weights have keys starting with "lora_te" or "lora_te1_text_model" etc.
            for key in lora_raw.keys():
                # Check if this is a CLIP/text encoder key
                if any(clip_prefix in key for clip_prefix in [
                    'lora_te', 'text_encoder', 'lora_te1_text_model', 'lora_te2_text_model',
                    'text_model', 'transformer.text_model'
                ]):
                    # Copy CLIP weight from original to new state dict
                    # This preserves the unmodified CLIP weights
                    new_state_dict[key] = lora_raw[key]

        print(f"Saving LoRA to {save_path}")
        comfy.utils.save_torch_file(new_state_dict, save_path)

        return {}
