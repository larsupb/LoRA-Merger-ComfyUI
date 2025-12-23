from .types import LORA_STACK


class LoRASelect:
    """
    Select one LoRA out of a LoRAStack by its index.
    Optionally accepts raw LoRA dict to preserve CLIP weights.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key_dicts": ("LoRAStack",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Index of the LoRA to select."}),
            },
            "optional": {
                "lora_raw_dict": ("LoRARawDict", {"tooltip": "Optional raw LoRA dict to preserve CLIP weights"}),
            },
        }

    RETURN_TYPES = ("LoRABundle",)
    FUNCTION = "select_lora"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = "Select one LoRA from stack by index. Preserves CLIP weights if raw dict is provided."

    def select_lora(self, key_dicts: LORA_STACK, index: int, lora_raw_dict: dict = None) -> (dict,):
        keys = list(key_dicts.keys())
        if index < 0 or index >= len(keys):
            raise IndexError(f"Index {index} out of range for LoRAStack with {len(keys)} items.")
        selected_key = keys[index]

        bundle = {
            "lora": key_dicts[selected_key],
            "strength_model": 1.0,
            "name": selected_key
        }

        # Add raw LoRA data if available (for preserving CLIP weights)
        if lora_raw_dict is not None and selected_key in lora_raw_dict:
            bundle["lora_raw"] = lora_raw_dict[selected_key]

        return (bundle,)
