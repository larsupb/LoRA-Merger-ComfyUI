import concurrent.futures
import copy
import io
import json
import logging
from typing import Dict, Optional, Any, List, Tuple

import cairosvg
import numpy as np
import torch
from PIL import Image
from lxml import etree
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ImmutableMap
from mergekit.io.tasks import GatherTensors
from mergekit.sparsify import SparsificationMethod, RescaleNorm, sparsify

from comfy.utils import ProgressBar  # Assuming this is thread-safe or replaced
from comfy.weight_adapter import LoRAAdapter

import wan_lora
import sd_lora

from .utility import load_as_comfy_lora


class LoRAAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora": ("LoRA",),
                "indicator": (["frobenius_norm", "sparsity"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "blocks_store")

    FUNCTION = "run"
    CATEGORY = "LoRA PowerMerge/Analytics"

    def run(self, model: ModelReference, clip: ModelReference, lora: Dict[str, Any], indicator):
        if 'lora' not in lora or lora['lora'] is None:
            lora['lora'] = load_as_comfy_lora(lora, model, clip)

        # Calculate indicators for each tensor
        layer_measures = self.calculate_all_measures(lora['lora'])

        # Generate SVG from block_dict
        svg, color_settings = self.generate_svg(layer_measures, indicator=indicator, size=512)
        # Generate image from SVG
        image = self.generate_image(svg)

        return image, color_settings

    def process_layer(self, layer_key, lora_adapter):
        try:
            block_names = sd_lora.detect_block_names(layer_key)
            if not block_names or "main_block" not in block_names:
                logging.info(f"Skipping layer {layer_key} as it does not match the expected pattern.")
                return layer_key, None

            alpha = lora_adapter.weights[2]
            up_weights = lora_adapter.weights[0]
            down_weights = lora_adapter.weights[1]
            if up_weights.dim() > 2 or down_weights.dim() > 2:
                delta_W = alpha * up_weights.squeeze((2, 3)) @ down_weights.squeeze((2, 3))
            else:
                delta_W = alpha * up_weights @ down_weights

            frobenius_norm, mean_frobenius = self.frobenius_norm(delta_W)
            sparsity = self.sparsity(delta_W, epsilon=1e-3)

            return layer_key, {
                "main_block": block_names["main_block"],
                "sub_block": block_names["sub_block"] if "sub_block" in block_names else None,
                "frobenius_norm": mean_frobenius,
                "sparsity": sparsity,
            }
        except Exception as e:
            logging.error(f"Error processing {layer_key}: {e}")
            return layer_key, None

    def calculate_all_measures(self, patch_dict):
        layer_measures = {}
        pbar = ProgressBar(len(patch_dict))

        # Threaded version
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.process_layer, layer_key, lora_adapter): layer_key
                for layer_key, lora_adapter in patch_dict.items()
            }

            for future in concurrent.futures.as_completed(futures):
                layer_key, result = future.result()
                if result is not None:
                    layer_measures[layer_key] = result
                pbar.update(1)

        return layer_measures

    def frobenius_norm(self, tensor: torch.Tensor) -> tuple[float, float]:
        """Calculate the Frobenius norm of a tensor."""
        frobenius_norm = torch.norm(tensor, p='fro').item()
        frobenius_norm_mean = frobenius_norm / tensor.numel()
        return frobenius_norm, frobenius_norm_mean

    def sparsity(self, tensor: torch.Tensor, epsilon: float) -> float:
        """Calculate the sparsity of a tensor."""
        if tensor.numel() == 0:
            return 0.0
        return (torch.sum(torch.abs(tensor) < epsilon).item() / tensor.numel()) * 100.0

    def generate_svg(self, layer_measures: Dict[str, Dict[str, Any]], indicator="frobenius_norm", size=512) -> \
            Tuple[str, Dict[str, str]]:
        # Group the layer_measures by their main_block and calculate an average indicator value for each block.
        block_info = self.group_measures(indicator, layer_measures)

        # Modify the SVG shapes with the block_info
        color_settings = self.calculate_color_settings(block_info, indicator)

        # Load the SVG template and apply the color schema
        svg = self.apply_color_schema(block_info, color_settings)

        return svg, color_settings

    def apply_color_schema(self, block_info, color_settings):
        # read svg template from file js/sdxl_unet.svg
        with open("custom_nodes/LoRA-Merger-ComfyUI/js/sdxl_unet.svg", "r") as f:
            svg_template = f.read()

        root = etree.fromstring(svg_template)
        # SVGs use namespaces
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        for key, value in block_info.items():
            # Loop through each block and replace the placeholders in the SVG template
            # Find the svg element where id matches "{key}.rect"
            element_id = f"{key}.rect"
            # Find element by id
            target = root.xpath(f'//*[@id="{element_id}"]', namespaces=ns)
            if target:
                # Replace the background (search for fill) with a color based on the value
                color = color_settings[key]
                # Change attribute
                elem = target[0]
                elem.set("fill", color)
            else:
                logging.warning(f"Block {key} not found in SVG template. Skipping.")
        # Print modified SVG to string
        svg_template = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')
        return svg_template

    def calculate_color_settings(self, block_info, indicator) -> Dict[str, str]:
        """Creates a dictionary with color settings for each block based on the indicator."""
        color_settings = {}
        indicator_values = [v[indicator] for v in block_info.values() if indicator in v]
        min_value = min(indicator_values) if indicator_values else 0
        max_value = max(indicator_values) if indicator_values else 1

        def interpolate_color(v, min_color=(255, 255, 255), max_color=(255, 0, 0)):
            """Interpolate color based on the value."""
            if max_value == min_value:
                return min_color
            ratio = (v - min_value) / (max_value - min_value)
            r = int(min_color[0] + ratio * (max_color[0] - min_color[0]))
            g = int(min_color[1] + ratio * (max_color[1] - min_color[1]))
            b = int(min_color[2] + ratio * (max_color[2] - min_color[2]))
            return f'rgb({r}, {g}, {b})'

        for key, value in block_info.items():
            max_color = (0, 0, 255) if value['block_type'] == "sub_block" else (255, 0, 0)
            color_settings[key] = interpolate_color(value[indicator], max_color=max_color)
        return color_settings

    def group_measures(self, indicator, layer_measures):
        # group the values of layer_measures by their main_block. Calculate an average indicator value for each block.#
        block_info = {}
        for layer_key, measures in layer_measures.items():
            main_block = measures["main_block"]
            sub_block = measures["sub_block"]
            value = measures[indicator]

            # Aggregate values for each block
            if main_block not in block_info:
                block_info[main_block] = {"block_type": "main_block", indicator: 0.}
            block_info[main_block][indicator] += value

            # If sub_block is present, also aggregate it
            if sub_block:
                if sub_block not in block_info:
                    block_info[sub_block] = {"block_type": "sub_block", indicator: 0.}
                block_info[sub_block][indicator] += value

        # Normalize the values by the number of occurrences
        for block, values in block_info.items():
            count = sum(1 for v in layer_measures.values() if v["main_block"] == block or v["sub_block"] == block)
            if count > 0:
                for key in values:
                    # if values[key] is a number, normalize it
                    if isinstance(values[key], (int, float)):
                        values[key] /= count
        return block_info

    def generate_image(self, svg):
        # Convert SVG string to PNG
        png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))

        # Load into PIL Image
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        # Convert to numpy and normalize
        image_np = np.array(image).astype(np.float32) / 255.0

        # Convert to torch tensor and add batch dimension (B, H, W, C)
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # shape: [1, H, W, C]

        # Return shape (B, H, W, C)
        return image_tensor


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
        widget_data = self.parse(blocks_store)
        architecture = widget_data.get("mode", "sdxl_unet")
        block_scale_dict = widget_data.get("blockScales", {})

        # Workaround for middle_block expected but middle_blocks supplied
        if "middle_blocks.1" in block_scale_dict:
            block_scale_dict["middle_block.1"] = block_scale_dict.pop("middle_blocks.1")
        print("Block scale dict:", block_scale_dict)

        new_key_dicts = {}
        for lora_name, patch_dict in key_dicts.items():
            patch_dict_modified = self.apply(patch_dict, block_scale_dict, architecture)
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

    def parse(self, stringified: str) -> Dict[str, Dict[str, Any]]:
        try:
            return json.loads(stringified)  # This will now be a proper JSON string
        except:
            print(f"Failed to parse JSON string: {stringified}.\n Returning empty dictionary.")
        return {}


class LoRAPrune:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LoRA",),
                "method": (["magnitude", "random", "magnitude_outliers", "della_magprune", "cosine_prune", "zscore", "iqr"],),
                "density": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.,
                    "max": 1.,
                    "step": 0.01,
                    "tooltip": "Fraction of weights to retain in the sparsified task vector",
                }),
                "gamma": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "magnitude_outliers: Fraction of the parameters with the largest absolute magnitudes "
                               "are identified for removal",
                }),
                "epsilon": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Della-MagPrune: Defines the half-width of the range for keep probabilities. "
                               "Keep probabilities for parameters in a row will range from density - epsilon to "
                               "density + epsilon, mapped from the smallest to largest magnitude parameters "
                               "in that row, respectively. epsilon must be chosen such that "
                               "density - epsilon > 0 and density + epsilon < 1.",
                }),
                "alpha_1": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.05,
                    "max": 4.55,
                    "step": 0.01,
                    "tooltip": "Similarity threshold for cosine prune.",
                }),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "prune"
    CATEGORY = "LoRA PowerMerge"

    def prune(self, lora, method: str = "iqr", density: float = 0.95,
              gamma: float = 0.05, epsilon: float = 0.05, alpha_1: float = 0.2):
        # check if lora is None or lora['lora'] is None
        if lora is None or lora.get('lora_raw') is None:
            raise ValueError("LoRA is not provided or is empty. Please provide a valid LoRA.")

        state_dict = lora['lora_raw']
        state_dict_pruned = {}
        pbar = ProgressBar(len(state_dict))
        for key, tensor in state_dict.items():
            # tensor dimensions must be 2D or 3D else skip
            if tensor.dim() < 2 or tensor.dim() > 3:
                state_dict_pruned[key] = tensor
                continue
            # Flatten for indexing
            a_flat = tensor.view(-1)

            # Outlier detection
            if method == 'iqr':
                a_flat = iqr(a_flat, [a_flat])
                merged = a_flat.view_as(tensor)
            elif method == 'zscore':
                a_flat = zscore(a_flat)
                merged = a_flat.view_as(tensor)
            elif method == 'cosine_prune':
                a_flat = cosine_prune(a_flat, alpha_1)
                merged = a_flat.view_as(tensor)
            else:
                # parse method and indentify SparsificationMethod
                if method not in SparsificationMethod.__members__:
                    raise ValueError(f"Invalid sparsification method: {method}")
                sparsification_method = SparsificationMethod[method]
                rescale_norm = False  # TODO: add rescale_norm input
                merged = sparsify(
                    tensor,
                    density=density,
                    gamma=gamma,
                    epsilon=epsilon,
                    method=sparsification_method,
                    rescale_norm=RescaleNorm.l1 if rescale_norm else None,
                )
            state_dict_pruned[key] = merged
            pbar.update(1)

        lora_out = {"lora_raw": state_dict_pruned, "strength_model": 1, "strength_clip": 1,
                    "name": "Merge"}
        return (lora_out,)


def iqr(merged_flat: torch.Tensor, flat_tensors: List[torch.Tensor]) -> torch.Tensor:
    print(f"IQR: Minimum value: {merged_flat.min().item()}, maximum value: {merged_flat.max().item()}")
    print(f"IQR:: Total: {merged_flat.sum().item()}, "
          f"mean: {merged_flat.mean().item()}, standard deviation: {merged_flat.std().item()}")
    # Compute Q1 and Q3
    q1 = merged_flat.quantile(0.25)
    q3 = merged_flat.quantile(0.75)
    print(f"IQR: Q1: {q1}, Q3: {q3}")

    # Compute IQR
    iqr = q3 - q1

    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f"IQR: Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Detect outliers
    outliers = (merged_flat < lower_bound) | (merged_flat > upper_bound)
    # Stack all tensors and compute the mean across tensors for each position
    stacked = torch.stack(flat_tensors)  # Shape: [num_tensors, num_elements]
    means = stacked.mean(dim=0)  # Shape: [num_elements]
    # Replace outliers with mean from all tensors at that position
    merged_flat[outliers] = means[outliers]

    print(f"Total outliers detected: {outliers.sum().item()} "
          f"Percentage: {outliers.sum().item() / merged_flat.numel() * 100:.2f}%")
    return merged_flat


def zscore(merged_flat: torch.Tensor) -> torch.Tensor:
    # Compute the mean and standard deviation
    mean = merged_flat.mean()
    std = merged_flat.std()

    # Compute the Z-scores
    z_scores = torch.abs(merged_flat - mean) / std

    # Define a threshold for outlier detection (e.g., 3 standard deviations)
    threshold = 3.0
    outliers = z_scores > threshold

    # Replace outliers with the mean (or any other strategy)
    merged_flat[outliers] = 0

    # Reshape back to original shape
    return merged_flat


def cosine_prune(a_flat: torch.Tensor, alpha_1: float = 0.2) -> torch.Tensor:
    # Compute difference and sort by descending diff
    _, indices = torch.sort(torch.abs(a_flat), descending=True)

    # Custom cosine weight function
    T = a_flat.numel()
    t = torch.arange(T, device=a_flat.device, dtype=a_flat.dtype)
    alpha_1 = alpha_1
    weight = ((1 + torch.cos(torch.pi * t / T)) / 2) ** alpha_1
    weight = weight[indices]

    # Interpolate: more difference → more of b
    merged_flat = a_flat.clone()
    merged_flat[indices] = a_flat[indices] * weight

    return merged_flat


class CosineMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha_1": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.05,
                    "max": 4.55,
                    "step": 0.001,
                    "tooltip": "Similarity threshold for Cosine merge.",
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized.",
                }),
                "lambda_": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Lambda value for scaling the merged model.",
                }),
            },
        }

    RETURN_TYPES = ("PMBaseModelMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge"

    def get_method(self, alpha_1: float = 0.2, outlier_detection: str = "iqr", normalize: bool = True,
                   lambda_: float = 1.):
        method_def = {
            "name": "cosine_merge",
            "settings": {
                "alpha_1": alpha_1,
                "outlier_detection": outlier_detection if outlier_detection != "none" else None,
                "lambda_": lambda_,
                "normalize": normalize
            }
        }
        return (method_def,)


def cosine_merge(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        base_model: ModelReference,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    method_args = method_args or {}

    if len(tensors) != 2:
        raise RuntimeError("Cosine Merge expects exactly two models")

    # Extract LoRA tensors
    keys = list(tensors.keys())
    a = tensors.pop(keys[0])  # first_model
    b = tensors.pop(keys[1])  # second_model

    # Flatten for indexing
    a_flat = a.view(-1)
    b_flat = b.view(-1)

    # Compute difference and sort by descending diff
    absdiff = torch.abs(a_flat - b_flat)
    _, indices = torch.sort(absdiff, descending=True)

    # Custom cosine weight function
    T = a.numel()
    t = torch.arange(T, device=a.device, dtype=a.dtype)
    alpha_1 = method_args.get("alpha_1", 0.2)
    weight = ((1 + torch.cos(torch.pi * t / T)) / 2) ** alpha_1
    weight = weight[indices]

    # Interpolate: more difference → more of b
    merged_flat = a_flat.clone()
    merged_flat[indices] = a_flat[indices] * (1 - weight) + b_flat[indices] * weight
    # Reshape back
    merged = merged_flat.view_as(a)

    # Normalize if needed
    divisor = 1.
    if method_args['normalize']:
        divisor = torch.tensor(2.)
        divisor[divisor.abs() < 1e-8] = 1

    return merged * method_args['lambda_'] / divisor
