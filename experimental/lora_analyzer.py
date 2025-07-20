import concurrent.futures
import io
import logging
from typing import Dict, Any, Tuple

import cairosvg
import numpy as np
import torch
from PIL import Image
from lxml import etree
from mergekit.common import ModelReference

from ..architectures import sd_lora
from comfy.utils import ProgressBar  # Assuming this is thread-safe or replaced
from ..utility import load_as_comfy_lora


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