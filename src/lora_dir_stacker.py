import os

from .merge import parse_layer_filter, apply_layer_filter
from .types import LORA_STACK, LORA_WEIGHTS


class LoraStackFromDir:
    """
       Node for loading LoRA weights
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "directory": ("STRING",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "General model strength applied to all LoRAs."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "General CLIP strength applied to all LoRAs."}),
                "layer_filter": (
                    ["full", "attn-mlp", "attn-only"], {"default": "full", "tooltip": "Filter for specific layers."}),
                "sort_by": (["name", "name descending", "date", "date descending"],
                            {"default": "name", "tooltip": "Sort LoRAs by name or size."}),
                "limit": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Limit the number of LoRAs to load."}),
            },
        }

    RETURN_TYPES = ("LoRAStack", "LoRAWeights", "LoRARawDict",)
    FUNCTION = "stack_loras"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = "Stacks LoRA weights from the given directory and applies them to the model."

    def stack_loras(self, model, directory, strength_model: float = 1.0, strength_clip: float = 1.0,
                    layer_filter=None, sort_by: str = None, limit: int = 0) -> \
            (LORA_STACK, LORA_WEIGHTS, dict):
        # check if directory exists
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")

        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

        layer_filter = parse_layer_filter(layer_filter)

        # Load LoRAs and patch key names
        lora_patch_dicts = {}
        lora_strengths = {}
        lora_raw_dicts = {}  # Store raw LoRA state dicts for CLIP weights

        # Load LoRAs from the directory
        # walk over files in the directory
        for root, _, files in os.walk(directory):
            # Sort files based on the specified criteria
            if sort_by == "name":
                files = sorted(files)
            elif sort_by == "name descending":
                files = sorted(files, reverse=True)
            elif sort_by == "date":
                files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f)))
            elif sort_by == "date descending":
                files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f)), reverse=True)
            # Limit the number of LoRAs to load
            if limit > 0:
                files = files[:limit]

            for file in files:
                if file.endswith(".safetensors") or file.endswith(".ckpt"):
                    lora_path = os.path.join(root, file)
                    lora_name = os.path.splitext(file)[0]
                    lora_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    patch_dict = comfy.lora.load_lora(lora_raw, key_map)
                    patch_dict = apply_layer_filter(patch_dict, layer_filter)
                    lora_patch_dicts[lora_name] = patch_dict
                    lora_strengths[lora_name] = {
                        'strength_model': strength_model,
                        'strength_clip': strength_clip,
                    }
                    lora_raw_dicts[lora_name] = lora_raw  # Store raw state dict

        return lora_patch_dicts, lora_strengths, lora_raw_dicts,
