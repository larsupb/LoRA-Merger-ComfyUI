import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import torch
import comfy
import comfy.model_management
from comfy.weight_adapter import LoRAAdapter

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ImmutableMap, ModelPath
from mergekit.io.tasks import GatherTensors

from .architectures.sd_lora import weights_as_tuple

# Import merge module components
from .merge import (
    create_map,
    create_tensor_param,
    get_merge_method,
    prepare_method_args,
)
from .mergekit_utils import load_on_device
# Import centralized types
from .types import (
    LORA_WEIGHTS,
    LORA_TENSOR_DICT,
    LORA_TENSORS_BY_LAYER,
    MergeMethod,
)
from .utility import map_device
# Import validation components
from .validation import validate_tensor_shapes_compatible


# Helper functions moved to src/merge/utils.py
# Imported above for backward compatibility


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
