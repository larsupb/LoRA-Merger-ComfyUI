import json
import logging
from typing import Dict, Any

import torch

from comfy.weight_adapter import LoRAAdapter
from .architectures import sd_lora, wan_lora

class LoRAModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_dicts": ("LoRAKeyDict", {"tooltip": "The dictionary containing LoRA names and key weights."}),
                "blocks_store": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("LoRAKeyDict",)

    FUNCTION = "run"
    CATEGORY = "LoRA PowerMerge"

    def run(self, key_dicts: sd_lora.LORAS_LORA_KEY_DICT, blocks_store: str):
        widget_data: dict = self.parse(blocks_store)
        arch: str = widget_data.get("mode", "sdxl_unet")
        block_scale_dict = widget_data.get("blockScales", {})

        # Workaround for middle_block expected but middle_blocks provided
        if "middle_blocks.1" in block_scale_dict:
            block_scale_dict["middle_block.1"] = block_scale_dict.pop("middle_blocks.1")
        print("Block scale dict:", block_scale_dict)
        # Workaround for WAN arch, blocks expected to start with index 0 instead of 1, so each key must be adjusted by -1
        if arch == "wan21":
            block_scale_dict = {f"{k.split('.')[0]}.{int(k.split('.')[1]) - 1}": v for k, v in block_scale_dict.items()}

        new_key_dicts = {}
        for lora_name, patch_dict in key_dicts.items():
            patch_dict_modified = self.apply(patch_dict, block_scale_dict, architecture=arch)
            new_key_dicts[lora_name] = patch_dict_modified

        return (new_key_dicts,)

    def apply(self, patch_dict : sd_lora.LORA_KEY_DICT, block_scale_dict: dict, architecture: str):
        # Iterate over keys in the LoRA adapter
        # Sum up the total weight of tensors for debugging
        total_weight = 0.0
        total_weight_after = 0.0
        patch_dict_filtered = {}
        for layer_key, adapter in patch_dict.items():
            total_weight += torch.sum(adapter.weights[0]).item() + torch.sum(adapter.weights[1]).item()

            # copy the weights to avoid modifying the original adapter
            new_weights = []
            for weight in adapter.weights:
                # copy if tensor
                if isinstance(weight, torch.Tensor):
                    new_weights.append(weight.clone())
                else:
                    new_weights.append(weight)

            block_names = sd_lora.detect_block_names(layer_key) if "sd" in architecture else wan_lora.detect_block_names(layer_key)
            if (block_names is None or "main_block" not in block_names
                    or block_names["main_block"] not in block_scale_dict):
                # Skip scaling for this layer
                logging.info(f"Skipping layer {layer_key} as it was not mentioned by the block scale dict.")
            else:
                scale_factor = float(block_scale_dict[block_names["main_block"]])
                # Apply the scale factor to the weights
                new_weights[0] *= scale_factor
                new_weights[1] *= scale_factor
            # Sum up the total weight of tensors for debugging
            total_weight_after += torch.sum(new_weights[0]).item() + torch.sum(new_weights[1]).item()

            patch_dict_filtered[layer_key] = LoRAAdapter(adapter.loaded_keys, new_weights)
        logging.info(f"Modified LoRA: {len(patch_dict_filtered)} layers after scaling.")
        logging.info(f"Total weight before scaling: {total_weight}, after scaling: {total_weight_after}")
        return patch_dict_filtered

    def parse(self, stringified: str) -> Dict[str, Any]:
        try:
            return json.loads(stringified)  # This will now be a proper JSON string
        except:
            print(f"Failed to parse JSON string: {stringified}.\n Returning empty dictionary.")
        return {}


