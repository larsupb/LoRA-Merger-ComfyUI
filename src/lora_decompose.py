import hashlib
import logging
import time
from typing import Optional, Set

import torch
import comfy

from .architectures.sd_lora import analyse_keys, calc_up_down_alphas
from .types import LORA_TENSORS_BY_LAYER, LORA_STACK, LORA_TENSOR_DICT
from .utility import map_device, adjust_tensor_dims


class LoraDecompose:
    """
       Node for decomposing LoRA models into their components
    """

    def __init__(self):
        self.last_lora_names_hash: Optional[list] = None
        self.last_tensor_sum: float = 0.0
        self.last_svd_rank: int = -1
        self.last_decomposition_method: str = ""
        self.last_layer_filter: Optional[Set[str]] = None
        self.last_result: LORA_TENSORS_BY_LAYER = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key_dicts": ("LoRAStack",),
                 "decomposition_method": (
                    ["none", "rSVD", "energy_rSVD", "SVD"],
                    {
                        "default": "rSVD",
                        "tooltip": (
                            "Method used to reconcile LoRA ranks when they differ. "
                            "'none' will raise an error if ranks do not match. "
                            "'SVD' uses full singular value decomposition (slow but optimal). "
                            "'rSVD' uses randomized SVD (much faster, near-optimal). "
                            "'energy_rSVD' first prunes low-energy LoRA components and then "
                            "applies randomized SVD for fast, stable rank reduction "
                            "(recommended for DiT and large LoRAs)."
                        ),
                    }
                ),
                "svd_rank": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 128,
                        "tooltip": (
                            "Target LoRA rank after decomposition. "
                            "-1 keeps the rank of the first LoRA. "
                            "Lower values reduce model size and strength."
                        ),
                    }
                ),
                "device": (["cuda", "cpu"],
                           {"tooltip": "Decomposition device. Note: All decomposition uses float32 internally for numerical stability, then converts back to the original dtype."}
                           ),
            },
        }

    RETURN_TYPES = ("LoRATensors",)
    FUNCTION = "lora_decompose"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = """Decomposes LoRA stack into tensor components for merging.

Extracts (up, down, alpha) tuples from each LoRA layer and handles rank mismatches using SVD-based decomposition methods.

Decomposition Methods:
- none: Requires all LoRAs to have matching ranks (fastest, fails if ranks differ)
- rSVD: Randomized SVD for rank reconciliation (fast, recommended for most cases)
- energy_rSVD: Energy-based randomized SVD (best for DiT/large LoRAs)
- SVD: Full SVD decomposition (slow but optimal)

Features hash-based caching to skip recomputation when inputs haven't changed."""

    def lora_decompose(self, key_dicts: LORA_STACK = None,
                       decomposition_method="rSVD", svd_rank=-1, device=None):
        device, _ = map_device(device, "float32")

        logging.info(f"Decomposing LoRAs with method: {decomposition_method}, SVD rank: {svd_rank}")

        # check if key_dicts differs from the previous one
        lora_names_hash_new = self.compute_hash(list(key_dicts.keys()))
        if (self.last_lora_names_hash == lora_names_hash_new
                and self.last_svd_rank == svd_rank
                and self.last_decomposition_method == decomposition_method
                and self.last_tensor_sum == self.compute_sum(key_dicts)):
            logging.info("Key dicts have not changed, returning last result.")
            if self.last_result is not None:
                return (self.last_result,)
            else:
                logging.warning("No last result available, recomputing.")
        else:
            logging.info("Key dicts have changed, recomputing.")

        self.last_lora_names_hash = lora_names_hash_new
        self.last_tensor_sum = self.compute_sum(key_dicts)
        self.last_svd_rank = svd_rank
        self.last_decomposition_method = decomposition_method

        self.last_result = self.decompose(key_dicts=key_dicts, device=device,
                                          decomposition_method=decomposition_method,
                                          svd_rank=svd_rank)
        return (self.last_result,)

    @staticmethod
    def compute_hash(value):
        """Computes a hash of the value for change detection."""
        return hashlib.md5(str(value).encode()).hexdigest()

    @staticmethod
    def compute_sum(lora_key_dicts: LORA_STACK):
        """Computes the sum of all up, down, and alpha tensors in the LoRA key dicts."""
        sum_ = 0
        for lora_name, lora_key_dict in lora_key_dicts.items():
            for key in lora_key_dict.keys():
                lora_adapter = lora_key_dict[key]
                up, down, _, _, _, _ = lora_adapter.weights
                sum_ += up.sum().item() + down.sum().item()
        return sum_

    def decompose(self, key_dicts, device, decomposition_method, svd_rank) -> LORA_TENSORS_BY_LAYER:
        """
        Decomposes LoRA models into their components.
        Args:
            key_dicts: Dictionary of LoRA names and their respective keys.
            device: Device to load tensors on.
            decomposition_method: Method to use for dimension alignment ("none", "svd", "rSVD", or "energy_rSVD").
            svd_rank: Target rank for decomposition.
        Returns:
            Dictionary of LoRA components.
            lora_key -> lora_name -> (up, down, alpha)
        """
        keys = list(analyse_keys(key_dicts))  # [:10]  # Limit to 100 keys for testing

        pbar = comfy.utils.ProgressBar(len(keys))
        start = time.time()

        def process_key(key, device_=device) -> LORA_TENSOR_DICT:
            uda = calc_up_down_alphas(key_dicts, key, load_device=device_, scale_to_alpha_0=True)

            # Determine if SVD should be applied
            if decomposition_method == "none":
                # Check if all LoRAs have the same rank
                ranks = [up.shape[1] for up, _, _ in uda.values()]
                if len(set(ranks)) > 1:
                    rank_info = {lora_name: up.shape[1] for lora_name, (up, _, _) in uda.items()}
                    raise ValueError(
                        f"LoRAs have different ranks for key '{key}': {rank_info}. "
                        f"Please select a decomposition method (SVD, rSVD, or energy_rSVD) to align dimensions."
                    )
                # No adjustment needed
                return uda
            else:
                # Apply the selected decomposition method
                uda_adjusted = adjust_tensor_dims(
                    uda,
                    apply_svd=True,
                    svd_rank=svd_rank,
                    method=decomposition_method
                )
                return uda_adjusted

        out = {}
        # Batch progress bar updates to reduce overhead for large layer counts (900+)
        update_frequency = max(1, len(keys) // 100)  # Update at most 100 times
        for i, key in enumerate(keys):
            out[key] = process_key(key)
            if (i + 1) % update_frequency == 0 or (i + 1) == len(keys):
                # Update by the batch size, or remaining items on last iteration
                batch_size = update_frequency if (i + 1) < len(keys) else ((i + 1) % update_frequency or update_frequency)
                pbar.update(batch_size)

        logging.info(f"Processed {len(keys)} keys in {time.time() - start:.2f} seconds")

        torch.cuda.empty_cache()

        return out
