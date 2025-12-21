import hashlib
import logging
import os
import time
import torch

import comfy
import comfy.model_management
from comfy.weight_adapter import LoRAAdapter

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Set
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ImmutableMap, ModelPath
from mergekit.io.tasks import GatherTensors

from .architectures.sd_lora import weights_as_tuple, analyse_keys, calc_up_down_alphas

# Import merge module components
from .merge import (
    create_map,
    create_tensor_param,
    parse_layer_filter,
    apply_layer_filter,
    get_merge_method,
    prepare_method_args,
)
from .mergekit_utils import load_on_device
# Import centralized types
from .types import (
    LORA_STACK,
    LORA_WEIGHTS,
    LORA_TENSOR_DICT,
    LORA_TENSORS_BY_LAYER,
    MergeMethod,
)
from .utility import map_device, adjust_tensor_dims
# Import validation components
from .validation import validate_tensor_shapes_compatible


# Helper functions moved to src/merge/utils.py
# Imported above for backward compatibility


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

    def stack_loras(self, model, directory, layer_filter=None, sort_by: str = None, limit: int = 0) -> \
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
                        'strength_model': 1.0,  # Default strength
                    }
                    lora_raw_dicts[lora_name] = lora_raw  # Store raw state dict

        return lora_patch_dicts, lora_strengths, lora_raw_dicts,


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
                "device": (["cuda", "cpu"],),
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
        for i, key in enumerate(keys):
            out[key] = process_key(key)
            pbar.update(1)

        logging.info(f"Processed {len(keys)} keys in {time.time() - start:.2f} seconds")

        torch.cuda.empty_cache()

        return out


class LoraMergerMergekit:
    """
       Node for merging LoRA models with Mergekit
    """

    def __init__(self):
        self.components: LORA_TENSORS_BY_LAYER = {}
        self.strengths: LORA_WEIGHTS = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": ("MergeMethod",),
                "components": ("LoRATensors", {"tooltip": "The decomposed components of the LoRAs to be merged."}),
                "strengths": ("LoRAWeights", {"tooltip": "The weights of the LoRAs to be merged."}),
                "lambda_": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Lambda value for scaling the merged model.",
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float16", "bfloat16", "float32"],),
            },
        }

    RETURN_TYPES = ("LoRABundle", "MergeContext")
    RETURN_NAMES = ("lora", "merge_context")
    FUNCTION = "lora_mergekit"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = """Core LoRA merger using Mergekit algorithms.

Merges decomposed LoRA components using the selected merge method (TIES, DARE, SLERP, etc.). Processes layers in parallel using ThreadPoolExecutor for performance.

Inputs:
- method: Merge algorithm configuration from method nodes
- components: Decomposed LoRA tensors from LoRA Decompose node
- strengths: Per-LoRA weight multipliers (strength_model for UNet, strength_clip for CLIP)
- lambda_: Global scaling factor applied to final merged result (0-1)

Outputs:
- LoRABundle: Merged LoRA ready for application or saving
- MergeContext: Reusable merge configuration for batch operations"""

    @torch.no_grad()
    def lora_mergekit(self,
                      method: Dict = None,
                      components: LORA_TENSORS_BY_LAYER = None,
                      strengths: LORA_WEIGHTS = None,
                      lambda_: float = 1.0,
                      device=None, dtype=None):

        if components is None:
            raise Exception("No components provided for merging.")

        self.components = components
        self.strengths = strengths

        device, dtype = map_device(device, dtype)

        # Use dispatcher to get merge method
        merge_method = get_merge_method(method['name'])

        # Prepare method arguments
        method_args = prepare_method_args(method['name'], method['settings'])

        self.validate_input()

        # Adjust components to match the method requirements
        merge = self.merge(method=merge_method, method_args=method_args, lambda_=lambda_, device=device, dtype=dtype)
        # Clean up VRAM
        torch.cuda.empty_cache()

        # Create merge context for downstream nodes
        merge_context = {
            "method": method,
            "components": components,
            "strengths": strengths,
            "lambda_": lambda_,
            "device": device,
            "dtype": dtype
        }

        return merge + (merge_context,)

    def merge(self, method: MergeMethod, method_args, lambda_, device, dtype):
        pbar = comfy.utils.ProgressBar(len(self.components.keys()))
        start = time.time()

        def process_key(key):
            lora_key_tuples: LORA_TENSOR_DICT = self.components[key]

            weights = [self.strengths[lora_name]["strength_model"] for lora_name in lora_key_tuples.keys()]
            weights = torch.tensor(weights, dtype=dtype).to(device=device)

            def calculate(tensors_):
                tensor_map = {}
                tensor_weight_map = {}
                weight_info = WeightInfo(name=f'{key}.merge', dtype=dtype, is_embed=False)
                for i, t in enumerate(tensors_):
                    ref = ModelReference(model=ModelPath(path=f'{key}.{i}'))
                    tensor_map[ref] = t
                    tensor_weight_map[ref] = weights[i]

                gather_tensors = GatherTensors(weight_info=create_map(key, tensor_map, dtype))
                tensor_parameters = ImmutableMap(
                    {r: ImmutableMap(create_tensor_param(tensor_weight_map[r], method_args)) for r in
                     tensor_map.keys()})

                # Load to the device
                load_on_device(tensor_map, tensor_weight_map, device, dtype)

                # Call the merge method
                out = method(tensor_map, gather_tensors, weight_info, tensor_parameters, method_args)

                # Apply lambda scaling
                if lambda_ < 1.0:
                    out = out * lambda_

                # Offload the result to CPU
                load_on_device(tensor_map, tensor_weight_map, "cpu", dtype)
                out = out.to(device='cpu', dtype=torch.float32)
                return out

            # Extract up and down tensors
            up_tensors = [u for u, _, _ in lora_key_tuples.values()]
            down_tensors = [d for _, d, _ in lora_key_tuples.values()]

            # Debug logging
            if len(up_tensors) > 0:
                logging.debug(f"Key {key}: up_tensors[0] shape = {up_tensors[0].shape}")
            if len(down_tensors) > 0:
                logging.debug(f"Key {key}: down_tensors[0] shape = {down_tensors[0].shape}")

            up = calculate(up_tensors)
            down = calculate(down_tensors)
            alpha_0 = next(iter(lora_key_tuples.values()))[2]

            # Debug logging for results
            logging.debug(f"Key {key}: merged up shape = {up.shape}, merged down shape = {down.shape}")

            # Sanity check: up should be (out_features, rank) and down should be (rank, in_features)
            # For LoRA to work, up.shape[1] should equal down.shape[0] (the rank dimension)
            # If they're swapped, we'll detect it here
            if len(up.shape) == 2 and len(down.shape) == 2:
                # Check if up and down appear to be swapped
                # up should have more elements in dim 0 than dim 1 (tall matrix)
                # down should have more elements in dim 1 than dim 0 (wide matrix)
                up_is_tall = up.shape[0] > up.shape[1]
                down_is_wide = down.shape[1] > down.shape[0]

                # If up is NOT tall or down is NOT wide, they might be swapped
                if not up_is_tall and down_is_wide:
                    # up appears to be a wide matrix, might need transpose
                    logging.warning(f"Key {key}: up tensor appears to be transposed (shape {up.shape}). This may cause issues.")
                if up_is_tall and not down_is_wide:
                    # down appears to be a tall matrix, might need transpose
                    logging.warning(f"Key {key}: down tensor appears to be transposed (shape {down.shape}). This may cause issues.")

            return key, (up, down, alpha_0)

        adapter_state_dict = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            keys = self.components.keys()

            # distribute the work across available devices
            futures = {executor.submit(process_key, key) for key in keys}

            for future in as_completed(futures):
                key, result = future.result()
                if result:
                    up, down, alpha_0 = result
                    adapter_state_dict[key] = LoRAAdapter(weights=weights_as_tuple(up, down, alpha_0),
                                                          loaded_keys=set(keys))
                pbar.update(1)

        logging.info(f"Processed {len(keys)} keys in {time.time() - start:.2f} seconds")

        lora_out = {"lora": adapter_state_dict, "strength_model": 1, "name": "Merge"}
        return (lora_out,)

    def validate_input(self):
        """
        Validate input parameters for merge operation.

        Performs comprehensive validation of:
        - Component tensors (shapes, compatibility)
        - Strength values (presence, reasonable ranges)

        Logs warnings for potential issues and raises exceptions for
        critical errors that would cause merge to fail.

        Raises:
            ValueError: If critical validation errors are found
        """
        errors = []
        warnings = []

        # Validate components exist and are not empty
        if not self.components:
            raise ValueError("No components provided for merging")

        if len(self.components) == 0:
            raise ValueError("Components dictionary is empty")

        # Validate strengths exist
        if not self.strengths:
            raise ValueError("No strengths provided for merging")

        # Validate tensor shapes are compatible
        validation_result = validate_tensor_shapes_compatible(self.components)

        # Log validation warnings
        for warning in validation_result["warnings"]:
            logging.warning(f"Validation warning: {warning}")
            warnings.append(warning)

        # Collect validation errors
        for error in validation_result["errors"]:
            error_msg = f"{error['code']}: {error['message']}"
            if error.get('location'):
                error_msg += f" (at {error['location']})"
            errors.append(error_msg)
            logging.error(f"Validation error: {error_msg}")

        # Validate all LoRAs in components have corresponding strengths
        lora_names = set()
        for layer_tensors in self.components.values():
            lora_names.update(layer_tensors.keys())

        for lora_name in lora_names:
            if lora_name not in self.strengths:
                error_msg = f"Missing strength for LoRA '{lora_name}'"
                errors.append(error_msg)
                logging.error(f"Validation error: {error_msg}")

        # Raise exception if critical errors found
        if errors:
            error_summary = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(
                f"Validation failed with {len(errors)} error(s):\n{error_summary}"
            )


# ============================================================================
# Algorithm implementations moved to src/merge/algorithms.py
# The functions below have been extracted to the merge module for better organization
# ============================================================================
