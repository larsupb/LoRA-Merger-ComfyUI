import json
import logging
from typing import Dict, Any

import torch

# init logging to log with a prefix
logging.basicConfig(level=logging.INFO, format='[LoRAModifier] %(message)s')

from comfy.weight_adapter import LoRAAdapter
from .architectures import sd_lora, dit_lora
from .architectures.general_architecture import LORA_STACK, LORA_KEY_DICT

class LoRAModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_dicts": ("LoRAStack", {"tooltip": "The dictionary containing LoRA names and key weights."}),
                "blocks_store": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("LoRAStack",)

    FUNCTION = "run"
    CATEGORY = "LoRA PowerMerge"

    def run(self, key_dicts: LORA_STACK, blocks_store: str):
        widget_data: dict = self.parse(blocks_store)
        arch: str = widget_data.get("mode", "sdxl_unet")
        block_scale_dict = widget_data.get("blockScales", {})

        # Workaround for middle_block expected but middle_blocks provided
        if "middle_blocks.1" in block_scale_dict:
            block_scale_dict["middle_block.1"] = block_scale_dict.pop("middle_blocks.1")
        print("Block scale dict:", block_scale_dict)

        new_key_dicts = {}
        for lora_name, patch_dict in key_dicts.items():
            patch_dict_modified = self.apply(patch_dict, block_scale_dict, architecture=arch)
            new_key_dicts[lora_name] = patch_dict_modified

        return (new_key_dicts,)

    def apply(self, patch_dict : LORA_KEY_DICT, block_scale_dict: dict, architecture: str):
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

            # Select the appropriate detect function based on architecture
            if "dit" in architecture:
                block_names = dit_lora.detect_block_names(layer_key)
            else:  # sd/sdxl
                block_names = sd_lora.detect_block_names(layer_key)
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

            # Convert list to tuple to match LoRAAdapter expectations
            # LoRAAdapter expects (loaded_keys, weights) where weights is a tuple
            patch_dict_filtered[layer_key] = LoRAAdapter(loaded_keys=adapter.loaded_keys, weights=tuple(new_weights))
        logging.info(f"Modified LoRA: {len(patch_dict_filtered)} layers after scaling.")
        logging.info(f"Total weight before scaling: {total_weight}, after scaling: {total_weight_after}")
        return patch_dict_filtered

    def parse(self, stringified: str) -> Dict[str, Any]:
        try:
            return json.loads(stringified)  # This will now be a proper JSON string
        except:
            print(f"Failed to parse JSON string: {stringified}.\n Returning empty dictionary.")
        return {}


