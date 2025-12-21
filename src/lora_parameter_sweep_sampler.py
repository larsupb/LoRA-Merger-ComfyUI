import logging
import copy
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

import comfy.utils
from comfy_extras.nodes_custom_sampler import SamplerCustom

from .lora_mergekit_merge import LoraMergerMergekit
from .types import MergeContext
from .utility import load_font


def parse_parameter_values(value_string: str) -> List[float]:
    """
    Parse parameter values from a string supporting multiple formats.

    Supported formats:
    - Linspace: "0.25 - 0.75 | 3" → [0.25, 0.5, 0.75]
    - Step: "0.25 - 0.75 : 0.25" → [0.25, 0.5, 0.75]
    - Explicit: "0.25, 0.5, 0.75" → [0.25, 0.5, 0.75]

    Args:
        value_string: String representation of parameter values

    Returns:
        List of float values

    Raises:
        ValueError: If the string format is invalid
    """
    value_string = value_string.strip()

    # Linspace format: "min - max | num_points"
    if '|' in value_string:
        try:
            range_part, num_part = value_string.split('|')
            range_part = range_part.strip()
            num_points = int(num_part.strip())

            if '-' not in range_part:
                raise ValueError(f"Invalid linspace format: '{value_string}'. Expected 'min - max | num_points'")

            parts = range_part.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid linspace range: '{range_part}'. Expected 'min - max'")

            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())

            if num_points < 1:
                raise ValueError(f"Number of points must be >= 1, got {num_points}")

            return np.linspace(min_val, max_val, num_points).tolist()
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse linspace format '{value_string}': {e}")

    # Step format: "min - max : step"
    elif ':' in value_string:
        try:
            range_part, step_part = value_string.split(':')
            range_part = range_part.strip()
            step = float(step_part.strip())

            if '-' not in range_part:
                raise ValueError(f"Invalid step format: '{value_string}'. Expected 'min - max : step'")

            parts = range_part.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid step range: '{range_part}'. Expected 'min - max'")

            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())

            if step <= 0:
                raise ValueError(f"Step must be > 0, got {step}")

            # Use arange and add a small epsilon to include max_val
            values = np.arange(min_val, max_val + step/2, step).tolist()
            return values
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse step format '{value_string}': {e}")

    # Explicit comma-separated format: "0.25, 0.5, 0.75"
    elif ',' in value_string:
        try:
            values = [float(v.strip()) for v in value_string.split(',')]
            if not values:
                raise ValueError("No values found")
            return values
        except ValueError as e:
            raise ValueError(f"Failed to parse comma-separated format '{value_string}': {e}")

    # Single value
    else:
        try:
            return [float(value_string)]
        except ValueError as e:
            raise ValueError(f"Failed to parse single value '{value_string}': {e}")


class LoRAParameterSweepSampler:
    """
    Sample images while sweeping through different merge parameter values.

    This node takes a merge context from LoRA Merger (Mergekit) and samples
    images across a range of parameter values for the selected merge method.
    Useful for comparing the effect of different parameter settings visually.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": (
                    "INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}
                ),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "merge_context": ("MergeContext", {
                    "tooltip": "Merge configuration from LoRA Merger (Mergekit) node"
                }),
                "parameter_name": ("STRING", {
                    "default": "t",
                    "tooltip": "Name of the parameter to sweep (e.g., 't' for SLERP, 'density' for TIES)"
                }),
                "parameter_values": ("STRING", {
                    "default": "0.25 - 0.75 | 3",
                    "tooltip": "Parameter values to test. Formats: '0.25 - 0.75 | 3' (linspace), '0.25 - 0.75 : 0.25' (step), '0.25, 0.5, 0.75' (explicit). Ignored for boolean parameters (auto uses [False, True])."
                }),
                "parameter_name_2": ("STRING", {
                    "default": "",
                    "tooltip": "Optional second parameter name for 2D sweep (leave empty for single parameter)"
                }),
                "parameter_values_2": ("STRING", {
                    "default": "",
                    "tooltip": "Second parameter values (same formats as parameter_values). Ignored for boolean parameters."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "image_grid")
    FUNCTION = "sample"
    CATEGORY = "LoRA PowerMerge/sampling"
    DESCRIPTION = """
    Sweep through different merge parameter values and sample images for comparison.

    Takes a merge context from the LoRA Merger node and varies one or two parameters
    across a range of values. Generates images for all parameter combinations and creates
    a labeled comparison grid.

    Single parameter mode:
    - SLERP: parameter_name="t", parameter_values="0.0 - 1.0 | 5"
    - TIES: parameter_name="density", parameter_values="0.5, 0.7, 0.9"
    - DARE: parameter_name="density", parameter_values="0.5 - 0.95 : 0.15"

    Boolean parameter mode:
    - parameter_name="normalize" (automatically uses [False, True])
    - parameter_values is ignored for boolean parameters

    Dual parameter mode (n × m sampling):
    - parameter_name="density", parameter_values="0.5, 0.7, 0.9"
    - parameter_name_2="k", parameter_values_2="16, 32, 64"
    - Generates 3 × 3 = 9 images in a 2D grid
    - Works with boolean + numeric or boolean + boolean combinations

    Maximum 64 images total. Progress tracking shows overall completion.
    """

    def sample(self, model, vae, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas,
               latent_image, merge_context: MergeContext, parameter_name: str, parameter_values: str,
               parameter_name_2: str = "", parameter_values_2: str = ""):

        # Validate parameters exist in method settings and check if they are boolean
        method_settings = merge_context['method'].get('settings', {})

        # Check if parameter_name is boolean
        if parameter_name not in method_settings:
            available_params = list(method_settings.keys())
            raise ValueError(
                f"Parameter '{parameter_name}' not found in merge method '{merge_context['method']['name']}'. "
                f"Available parameters: {available_params if available_params else 'none (method has no configurable parameters)'}. "
                f"Please check the merge method documentation for valid parameter names."
            )

        is_param1_bool = isinstance(method_settings[parameter_name], bool)

        # Parse parameter values or use [False, True] for boolean parameters
        if is_param1_bool:
            param_values = [False, True]
            logging.info(f"PM LoRAParameterSweepSampler: Parameter '{parameter_name}' is boolean, using [False, True]")
        else:
            try:
                param_values = parse_parameter_values(parameter_values)
            except ValueError as e:
                raise ValueError(f"Invalid parameter_values format: {e}")

        # Check if dual parameter mode is enabled
        dual_mode = parameter_name_2.strip() != ""
        param_values_2 = []
        is_param2_bool = False

        if dual_mode:
            # Validate second parameter exists
            if parameter_name_2 not in method_settings:
                available_params = list(method_settings.keys())
                raise ValueError(
                    f"Parameter '{parameter_name_2}' not found in merge method '{merge_context['method']['name']}'. "
                    f"Available parameters: {available_params if available_params else 'none (method has no configurable parameters)'}. "
                    f"Please check the merge method documentation for valid parameter names."
                )

            is_param2_bool = isinstance(method_settings[parameter_name_2], bool)

            # Parse parameter values or use [False, True] for boolean parameters
            if is_param2_bool:
                param_values_2 = [False, True]
                logging.info(f"PM LoRAParameterSweepSampler: Parameter '{parameter_name_2}' is boolean, using [False, True]")
            else:
                try:
                    param_values_2 = parse_parameter_values(parameter_values_2)
                except ValueError as e:
                    raise ValueError(f"Invalid parameter_values_2 format: {e}")

        # Calculate total images and validate limit
        total_images = len(param_values) * len(param_values_2) if dual_mode else len(param_values)
        if total_images > 64:
            raise ValueError(
                f"Total number of images ({total_images}) exceeds maximum allowed (64). "
                f"{'Reduce parameter value counts or use single parameter mode.' if dual_mode else 'Reduce the number of parameter values.'}"
            )

        # Log sweep configuration
        if dual_mode:
            param1_type = "boolean" if is_param1_bool else "numeric"
            param2_type = "boolean" if is_param2_bool else "numeric"
            logging.info(f"PM LoRAParameterSweepSampler: 2D sweep of '{parameter_name}' ({param1_type}, {len(param_values)} values) × "
                        f"'{parameter_name_2}' ({param2_type}, {len(param_values_2)} values) = {total_images} total images")
            logging.info(f"  {parameter_name}: {param_values}")
            logging.info(f"  {parameter_name_2}: {param_values_2}")
        else:
            param_type = "boolean" if is_param1_bool else "numeric"
            logging.info(f"PM LoRAParameterSweepSampler: Sweeping {param_type} parameter '{parameter_name}' "
                        f"across {len(param_values)} values: {param_values}")

        # Create merger instance for performing merges
        merger = LoraMergerMergekit()
        ksampler = SamplerCustom()

        # Initialize overall progress bar
        pbar = comfy.utils.ProgressBar(total_images)

        latents_out = []
        images_out = []

        # Sample for each parameter value (or parameter value combination in dual mode)
        if dual_mode:
            # Dual parameter mode: nested loops for n × m sampling
            for i, param_value in enumerate(param_values):
                for j, param_value_2 in enumerate(param_values_2):
                    sample_num = i * len(param_values_2) + j + 1

                    # Create a deep copy of merge context and update both parameters
                    context = copy.deepcopy(merge_context)
                    if 'settings' not in context['method']:
                        context['method']['settings'] = {}
                    context['method']['settings'][parameter_name] = param_value
                    context['method']['settings'][parameter_name_2] = param_value_2

                    logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Merging with "
                                f"{parameter_name}={param_value}, {parameter_name_2}={param_value_2}")

                    # Perform merge with updated parameters
                    merged_lora, _ = merger.lora_mergekit(
                        method=context['method'],
                        components=context['components'],
                        strengths=context['strengths'],
                        lambda_=context['lambda_'],
                        device=context['device'],
                        dtype=context['dtype']
                    )

                    # Apply merged LoRA to model
                    new_model_patcher = model.clone()
                    new_model_patcher.add_patches(merged_lora['lora'], merged_lora['strength_model'])

                    # Sample image
                    logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Sampling image")
                    denoised, _ = ksampler.sample(
                        model=new_model_patcher,
                        add_noise=add_noise,
                        noise_seed=noise_seed,
                        cfg=cfg,
                        positive=positive,
                        negative=negative,
                        sampler=sampler,
                        sigmas=sigmas,
                        latent_image=latent_image,
                    )

                    latents_out.append(denoised)

                    # Decode to image for annotation
                    image = vae.decode(denoised['samples'])
                    if len(image.shape) == 5:
                        image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
                    image = image.squeeze(0)  # Remove batch dimension

                    # Annotate image with both parameters
                    annotated_image = self.annotate_image(parameter_name, param_value, image,
                                                         parameter_name_2, param_value_2)
                    images_out.append(annotated_image)

                    # Update overall progress
                    pbar.update(1)
        else:
            # Single parameter mode (backward compatible)
            for i, param_value in enumerate(param_values):
                sample_num = i + 1

                # Create a deep copy of merge context and update the parameter
                context = copy.deepcopy(merge_context)
                if 'settings' not in context['method']:
                    context['method']['settings'] = {}
                context['method']['settings'][parameter_name] = param_value

                logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Merging with {parameter_name}={param_value}")

                # Perform merge with updated parameter
                merged_lora, _ = merger.lora_mergekit(
                    method=context['method'],
                    components=context['components'],
                    strengths=context['strengths'],
                    lambda_=context['lambda_'],
                    device=context['device'],
                    dtype=context['dtype']
                )

                # Apply merged LoRA to model
                new_model_patcher = model.clone()
                new_model_patcher.add_patches(merged_lora['lora'], merged_lora['strength_model'])

                # Sample image
                logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Sampling image")
                denoised, _ = ksampler.sample(
                    model=new_model_patcher,
                    add_noise=add_noise,
                    noise_seed=noise_seed,
                    cfg=cfg,
                    positive=positive,
                    negative=negative,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=latent_image,
                )

                latents_out.append(denoised)

                # Decode to image for annotation
                image = vae.decode(denoised['samples'])
                if len(image.shape) == 5:
                    image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
                image = image.squeeze(0)  # Remove batch dimension

                # Annotate image
                annotated_image = self.annotate_image(parameter_name, param_value, image)
                images_out.append(annotated_image)

                # Update overall progress
                pbar.update(1)

        # Create grid (2D if dual mode, horizontal if single mode)
        if images_out:
            if dual_mode:
                grid_image = self.create_grid(images_out, rows=len(param_values))
            else:
                grid_image = self.create_grid(images_out)
        else:
            grid_image = torch.tensor([])

        # Stack latents
        latents_stacked = {
            "samples": torch.cat([l['samples'] for l in latents_out], dim=0)
        }

        logging.info(f"PM LoRAParameterSweepSampler: Generated {total_images} images")

        return latents_stacked, grid_image

    @staticmethod
    def annotate_image(parameter_name: str, parameter_value: float, img_tensor: torch.Tensor,
                      parameter_name_2: str = None, parameter_value_2: float = None) -> torch.Tensor:
        """
        Annotate an image with parameter information.

        Args:
            parameter_name: Name of the first parameter
            parameter_value: Value of the first parameter
            img_tensor: Image tensor (H, W, C) in 0-1 range
            parameter_name_2: Optional name of the second parameter
            parameter_value_2: Optional value of the second parameter

        Returns:
            Annotated image tensor with title prepended
        """
        # Load font with increased size (96 instead of 48 for better readability)
        from PIL import ImageFont
        import logging
        import os

        # Font is in repository root/fonts/, not src/fonts/
        font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts", "ShareTechMono-Regular.ttf")
        try:
            title_font = ImageFont.truetype(font_path, size=48)
        except OSError:
            logging.warning(f"PM LoRAParameterSweepSampler: Font not found at {font_path}, using default font.")
            title_font = ImageFont.load_default()

        # Convert tensor to PIL Image
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Create title text
        if parameter_name_2 is not None and parameter_value_2 is not None:
            # Dual parameter mode
            title = f"{parameter_name}: {parameter_value:.3f}, {parameter_name_2}: {parameter_value_2:.3f}"
        else:
            # Single parameter mode
            title = f"{parameter_name}: {parameter_value:.3f}"

        # Calculate title dimensions with increased padding
        title_bbox = title_font.getbbox(title)
        title_width = title_bbox[2] - title_bbox[0]
        title_padding = 12  # Increased from 6
        line_height = title_font.getbbox("A")[3] + title_padding
        title_text_height = line_height + title_padding

        # Create title image
        title_text_image = Image.new('RGB', (img.width, title_text_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(title_text_image)
        draw.text(
            ((img.width - title_width) // 2, title_padding // 2),
            title,
            font=title_font,
            fill=(255, 255, 255)
        )

        # Convert title to tensor
        title_text_image_tensor = torch.tensor(np.array(title_text_image).astype(np.float32) / 255.0)

        # Concatenate title and image vertically
        out_image = torch.cat([title_text_image_tensor, img_tensor], 0)

        return out_image

    @staticmethod
    def create_grid(images: List[torch.Tensor], rows: int = 1) -> torch.Tensor:
        """
        Create a grid from a list of images.

        Args:
            images: List of image tensors (H, W, C)
            rows: Number of rows in the grid (default: 1 for horizontal strip)

        Returns:
            Grid image tensor (1, H*rows, W*cols, C)
        """
        if not images:
            return torch.tensor([])

        num_images = len(images)
        img_height = images[0].shape[0]
        img_width = images[0].shape[1]
        num_channels = images[0].shape[2]

        # Calculate grid dimensions
        cols = num_images // rows if rows > 0 else num_images
        if num_images % rows != 0:
            cols += 1

        # Create grid (rows × cols)
        grid = torch.zeros(1, img_height * rows, img_width * cols, num_channels)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            # Calculate pixel positions
            row_start = row * img_height
            row_end = (row + 1) * img_height
            col_start = col * img_width
            col_end = (col + 1) * img_width

            grid[:, row_start:row_end, col_start:col_end, :] = img

        return grid
