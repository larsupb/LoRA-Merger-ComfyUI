import logging
import time

import comfy.utils
import torch
from comfy.weight_adapter import LoRAAdapter

from .types import LoRABundleDict
from .utility import map_device, adjust_tensor_dims


class LoraResizer:
    def __init__(self):
        self.loaded_lora = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LoRABundle",),
                "decomposition_method": (
                    ["rSVD", "energy_rSVD", "SVD"],
                    {
                        "default": "rSVD",
                        "tooltip": (
                            "Method used to reconcile LoRA ranks when they differ. "
                            "'SVD' uses full singular value decomposition (slow but optimal). "
                            "'rSVD' uses randomized SVD (much faster, near-optimal). "
                            "'energy_rSVD' first prunes low-energy LoRA components and then "
                            "applies randomized SVD for fast, stable rank reduction "
                            "(recommended for DiT and large LoRAs)."
                        ),
                    }
                ),
                "new_rank": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 128,
                        "tooltip": (
                            "Target LoRA rank after decomposition. "                            
                            "Lower values reduce model size and strength."
                        ),
                    }
                ),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRABundle",)
    FUNCTION = "lora_resize"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = """Resizes a LoRA to a different rank using tensor decomposition.

This node reduces or increases the rank of all layers in a LoRA model using SVD-based methods.
Lower ranks reduce memory usage and may reduce strength, while maintaining semantic meaning.

Decomposition Methods:
- SVD: Full singular value decomposition (slow but optimal)
- rSVD: Randomized SVD (fast, recommended for most cases)
- energy_rSVD: Energy-based randomized SVD (best for DiT/large LoRAs)

The resizing uses asymmetric singular value distribution (all S values in up matrix)
which differs from the symmetric distribution used in lora_decompose."""

    def lora_resize(
        self,
        lora: LoRABundleDict,
        decomposition_method: str = "rSVD",
        new_rank: int = 16,
        device: str = "cuda",
        dtype: str = "float32"
    ) -> tuple:
        """
        Resize a LoRA to a new rank using tensor decomposition.

        Args:
            lora: LoRA bundle containing lora_raw and lora (LoRAAdapter dict)
            decomposition_method: Method to use ('SVD', 'rSVD', 'energy_rSVD')
            new_rank: Target rank for all layers
            device: Device for computation
            dtype: Data type for computation

        Returns:
            Tuple containing resized LoRA bundle
        """
        device, dtype = map_device(device, dtype)

        logging.info(f"Resizing LoRA '{lora.get('name', 'unknown')}' to rank {new_rank} using {decomposition_method}")

        # Extract the LoRA adapter dictionary (layer_key -> LoRAAdapter)
        lora_adapters = lora["lora"]
        lora_raw = lora.get("lora_raw", {})

        # Get all keys from the LoRA
        keys = list(lora_adapters.keys())

        logging.info(f"Processing {len(keys)} layers")

        pbar = comfy.utils.ProgressBar(len(keys))
        start = time.time()

        # Process each layer
        resized_adapters = {}

        for key in keys:
            adapter = lora_adapters[key]
            up, down, alpha, mid, dora_scale, reshape = adapter.weights

            # Skip if mid exists (LoHA/LoCon - not supported for now)
            if mid is not None:
                logging.warning(f"Skipping layer {key}: LoHA/LoCon format with mid tensor not supported")
                resized_adapters[key] = adapter
                pbar.update(1)
                continue

            # Skip if DoRA scale exists (DoRA not fully supported)
            if dora_scale is not None:
                logging.warning(f"Skipping layer {key}: DoRA format not supported")
                resized_adapters[key] = adapter
                pbar.update(1)
                continue

            # Get current rank from down tensor
            current_rank = down.shape[0]

            # Handle alpha=None: in standard LoRA, None means alpha equals rank
            if alpha is None:
                alpha = current_rank

            # Check if resizing is needed
            if current_rank == new_rank:
                # No resizing needed
                resized_adapters[key] = adapter
                pbar.update(1)
                continue

            # Create a single-item dictionary for adjust_tensor_dims
            # adjust_tensor_dims expects Dict[str, LORA_TENSORS]
            temp_dict = {"temp": (up, down, alpha)}

            # Apply resizing using adjust_tensor_dims
            if decomposition_method == "none":
                # If method is 'none' and ranks differ, this will raise an error
                if current_rank != new_rank:
                    raise ValueError(
                        f"Layer '{key}' has rank {current_rank} but target is {new_rank}. "
                        f"Cannot resize with decomposition_method='none'. "
                        f"Please select a decomposition method (SVD, rSVD, or energy_rSVD)."
                    )
                resized_adapters[key] = adapter
            else:
                # Resize using the selected method
                resized_dict = adjust_tensor_dims(
                    temp_dict,
                    apply_svd=True,
                    svd_rank=new_rank,
                    method=decomposition_method
                )

                up_new, down_new, alpha_new = resized_dict["temp"]

                # Scale alpha proportionally to maintain the same effective strength
                # Original strength: (alpha / current_rank) * (up @ down)
                # New strength: (alpha_scaled / new_rank) * (up_new @ down_new)
                # To maintain same strength: alpha_scaled = alpha * (new_rank / current_rank)
                # However, since we want to preserve the original alpha value semantics,
                # we keep alpha unchanged and the SVD already captured the strength in the tensors
                # But if alpha was None (now set to current_rank), scale it to new_rank
                if alpha_new == current_rank:
                    alpha_new = new_rank

                # Log shapes and alpha for debugging
                # logging.info(f"Layer {key}: Original shapes up={up.shape}, down={down.shape}, rank={current_rank}, alpha={alpha}")
                # logging.info(f"Layer {key}: Resized shapes up_new={up_new.shape}, down_new={down_new.shape}, rank={new_rank}, alpha_new={alpha_new}")

                # Ensure tensors are contiguous (required for safetensors saving)
                # SVD operations can produce non-contiguous tensors
                up_new = up_new.contiguous()
                down_new = down_new.contiguous()

                # Generate the loaded_keys that will be used during save
                # These keys need to match the format that convert_to_regular_lora expects
                key_str = key[0] if isinstance(key, tuple) else key

                # Strip .weight suffix if present (matches convert_to_regular_lora logic)
                key_base = key_str.replace("diffusion_model.", "")
                if key_base.endswith(".weight"):
                    key_base = key_base[:-len(".weight")]

                # Convert dots to underscores for LoRA naming convention
                key_suffix = key_base.replace(".", "_")

                # Create the loaded_keys set for this adapter
                # Format matches what convert_to_regular_lora will generate
                new_loaded_keys = {
                    f"lora_unet_{key_suffix}.lora_up.weight",
                    f"lora_unet_{key_suffix}.lora_down.weight",
                    f"lora_unet_{key_suffix}.alpha"
                }

                # Create new LoRAAdapter with resized tensors
                # Keep mid, dora_scale as None since we skipped those cases
                # Set reshape to None - the tensors are already in the correct shape after SVD
                # and keeping the old reshape metadata would cause mismatches
                resized_adapters[key] = LoRAAdapter(
                    weights=(up_new, down_new, alpha_new, None, None, None),
                    loaded_keys=new_loaded_keys
                )

            pbar.update(1)

        logging.info(f"Resized {len(keys)} layers in {time.time() - start:.2f} seconds")

        # Create output bundle
        # Preserve the original raw state dict and metadata
        lora_out = {
            "lora_raw": lora_raw,  # Keep original raw dict (includes CLIP weights)
            "lora": resized_adapters,  # Resized adapters
            "strength_model": lora.get("strength_model", 1.0),
            "strength_clip": lora.get("strength_clip", 1.0),
            "name": f"{lora.get('name', 'LoRA')}_r{new_rank}"
        }

        torch.cuda.empty_cache()

        return (lora_out,)
