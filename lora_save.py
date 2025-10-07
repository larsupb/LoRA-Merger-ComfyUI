import comfy
import folder_paths
import math
import os


class LoraSave:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LoRA",),
                "file_name": ("STRING", {"multiline": False, "default": "merged"}),
                "extension": (["safetensors"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "lora_save"
    CATEGORY = "LoRA PowerMerge"
    OUTPUT_NODE = True

    def lora_save(self, lora, file_name, extension):
        """
        Saves a LoRA model, applying strength scaling to its layers.
        This version supports both 'lora_up'/'lora_down' and 'lora_A'/'lora_B' naming conventions.
        """
        # Define the save path for the LoRA file
        save_path = os.path.join(folder_paths.get_folder_paths("loras")[0], f"{file_name}.{extension}")

        # If model and clip strengths are neutral (1.0), save the original LoRA state dict
        if lora.get("strength_model", 1.0) == 1.0 and lora.get("strength_clip", 1.0) == 1.0:
            new_state_dict = lora["lora"]
        else:
            # If strengths are applied, create a new state dict and scale the weights
            new_state_dict = {}
            for key, tensor in lora["lora"].items():
                # Determine the scale based on whether the key is for the text encoder (clip) or the model
                scale = lora.get("strength_clip", 1.0) if "lora_te" in key else lora.get("strength_model", 1.0)

                # Calculate the square root of the scale for distribution between up/down layers
                sqrt_scale = math.sqrt(abs(scale))
                sign_scale = 1 if scale >= 0 else -1

                # Apply scaling based on the layer type (up/B or down/A)
                if "lora_up" in key or "lora_B" in key:
                    # Apply full scaling effect to the 'up' or 'B' layer
                    new_state_dict[key] = tensor * sqrt_scale * sign_scale
                elif "lora_down" in key or "lora_A" in key:
                    # Apply partial scaling to the 'down' or 'A' layer
                    new_state_dict[key] = tensor * sqrt_scale
                else:
                    # For any other keys, copy the tensor without modification
                    new_state_dict[key] = tensor

        print(f"Saving LoRA to {save_path}")
        # Save the processed state dictionary to the specified file path
        comfy.utils.save_torch_file(new_state_dict, save_path)

        # This is an output node, so it returns an empty dictionary
        return {}


# A dictionary that ComfyUI uses to map node names to their classes
NODE_CLASS_MAPPINGS = {
    "LoraSave": LoraSave
}

# A dictionary that ComfyUI uses to display node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraSave": "Lora Save"
}
