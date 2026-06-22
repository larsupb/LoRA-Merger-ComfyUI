import logging
import copy
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from .comfy_util import rebuild_guider_with_patches
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
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
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

    def sample(self, vae, noise, guider, sampler, sigmas, latent_image,
               merge_context: MergeContext, parameter_name: str, parameter_values: str,
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

        # Validate parameter combinations and filter invalid ones
        method_name = merge_context['method']['name']
        valid_combinations = []
        filtered_combinations = []

        if dual_mode:
            # Validate all parameter combinations
            for param_value in param_values:
                for param_value_2 in param_values_2:
                    # Create test settings dict
                    test_settings = copy.deepcopy(method_settings)
                    test_settings[parameter_name] = param_value
                    test_settings[parameter_name_2] = param_value_2

                    # Validate using MergeParameterValidator
                    from .validation import MergeParameterValidator
                    validation_result = MergeParameterValidator.validate_method_args(
                        method_name, test_settings
                    )

                    if validation_result["valid"]:
                        valid_combinations.append((param_value, param_value_2))
                    else:
                        filtered_combinations.append((param_value, param_value_2, validation_result["errors"]))
        else:
            # Validate single parameter values
            for param_value in param_values:
                # Create test settings dict
                test_settings = copy.deepcopy(method_settings)
                test_settings[parameter_name] = param_value

                # Validate using MergeParameterValidator
                from .validation import MergeParameterValidator
                validation_result = MergeParameterValidator.validate_method_args(
                    method_name, test_settings
                )

                if validation_result["valid"]:
                    valid_combinations.append(param_value)
                else:
                    filtered_combinations.append((param_value, validation_result["errors"]))

        # Log filtered combinations
        if filtered_combinations:
            logging.warning(f"\n{'='*80}")
            logging.warning(f"PM LoRAParameterSweepSampler: {len(filtered_combinations)} invalid parameter combination(s) filtered out:")
            logging.warning(f"{'='*80}")
            for combo in filtered_combinations:
                if dual_mode:
                    param_val, param_val_2, errors = combo
                    logging.warning(f"  ✗ {parameter_name}={param_val}, {parameter_name_2}={param_val_2}")
                else:
                    param_val, errors = combo
                    logging.warning(f"  ✗ {parameter_name}={param_val}")
                for error in errors:
                    logging.warning(f"      → {error['message']}")
            logging.warning(f"{'='*80}\n")

        # Check if we have any valid combinations
        if not valid_combinations:
            raise ValueError(
                f"All parameter combinations are invalid! Check the console warnings above for details. "
                f"Adjust your parameter ranges to ensure at least one valid combination."
            )

        # Update param_values with only valid combinations
        if dual_mode:
            # Rebuild param_values and param_values_2 from valid combinations
            param_values = sorted(list(set(combo[0] for combo in valid_combinations)))
            param_values_2 = sorted(list(set(combo[1] for combo in valid_combinations)))
            # Keep only valid combinations
            valid_combination_set = set(valid_combinations)
        else:
            param_values = valid_combinations

        # Calculate total images and validate limit
        total_images = len(valid_combinations)
        if total_images > 64:
            raise ValueError(
                f"Total number of valid images ({total_images}) exceeds maximum allowed (64). "
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

        # Single progress bar spanning every sampling step across all images, with live
        # previews — same approach as the Block/Stack samplers (instead of a coarse
        # per-image bar plus an inner step bar that resets each image).
        steps = sigmas.shape[-1] - 1
        total_steps = total_images * steps
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        previewer = latent_preview.get_previewer(
            guider.model_patcher.load_device, guider.model_patcher.model.latent_format
        )
        pbar = comfy.utils.ProgressBar(total_steps) if not disable_pbar else None
        x0_holder = {}

        def make_callback(image_idx):
            img_start = image_idx * steps

            def cb(step, x0, x, total_steps_inner):
                x0_holder["x0"] = x0
                if pbar is not None:
                    preview_bytes = None
                    if previewer is not None:
                        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    pbar.update_absolute(img_start + step + 1, total_steps, preview_bytes)

            return cb

        latents_out = []
        images_out = []

        # Sample for each parameter value (or parameter value combination in dual mode)
        if dual_mode:
            sample_num = 0
            for param_value, param_value_2 in valid_combinations:
                sample_num += 1

                context = copy.deepcopy(merge_context)
                if 'settings' not in context['method']:
                    context['method']['settings'] = {}
                context['method']['settings'][parameter_name] = param_value
                context['method']['settings'][parameter_name_2] = param_value_2

                logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Merging with "
                            f"{parameter_name}={param_value}, {parameter_name_2}={param_value_2}")

                merged_lora, _ = merger.lora_mergekit(
                    method=context['method'],
                    components=context['components'],
                    strengths=context['strengths'],
                    lambda_=context['lambda_'],
                    device=context['device'],
                    dtype=context['dtype']
                )

                logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Sampling image")
                denoised = self.do_sample(
                    guider, merged_lora['lora'], merged_lora['strength_model'],
                    latent_image, noise, sampler, sigmas,
                    callback=make_callback(sample_num - 1), x0_output=x0_holder,
                    disable_pbar=disable_pbar
                )

                latents_out.append(denoised)

                # vae.decode returns: (B, H, W, C) channels-last; fix iteration for B=1
                images = vae.decode(denoised['samples'])  # (B, H, W, C)
                if len(images.shape) == 5:
                    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                if images.shape[0] > 1:
                    images = images.squeeze(0)
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                # images: (B, H, W, C)
                for img in images:
                    # img: (H, W, C) channels-last
                    annotated_image = self.annotate_image(parameter_name, param_value, img,
                                                         parameter_name_2, param_value_2)
                    images_out.append(annotated_image)
        else:
            for i, param_value in enumerate(param_values):
                sample_num = i + 1

                context = copy.deepcopy(merge_context)
                if 'settings' not in context['method']:
                    context['method']['settings'] = {}
                context['method']['settings'][parameter_name] = param_value

                logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Merging with {parameter_name}={param_value}")

                merged_lora, _ = merger.lora_mergekit(
                    method=context['method'],
                    components=context['components'],
                    strengths=context['strengths'],
                    lambda_=context['lambda_'],
                    device=context['device'],
                    dtype=context['dtype']
                )

                logging.info(f"PM LoRAParameterSweepSampler: [{sample_num}/{total_images}] Sampling image")
                denoised = self.do_sample(
                    guider, merged_lora['lora'], merged_lora['strength_model'],
                    latent_image, noise, sampler, sigmas,
                    callback=make_callback(i), x0_output=x0_holder,
                    disable_pbar=disable_pbar
                )

                latents_out.append(denoised)

                images = vae.decode(denoised['samples'])  # (B, H, W, C)
                if len(images.shape) == 5:
                    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                if images.shape[0] > 1:
                    images = images.squeeze(0)
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                for img in images:
                    annotated_image = self.annotate_image(parameter_name, param_value, img)
                    images_out.append(annotated_image)

        # Create grid (2D if dual mode, horizontal if single mode)
        if images_out:
            if dual_mode:
                # Calculate rows based on actual valid combinations
                # Try to make grid as square as possible
                import math
                num_images = len(images_out)
                rows = int(math.sqrt(num_images))
                grid_image = self.create_grid(images_out, rows=rows)
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
    def do_sample(guider, lora_patch_dict, lora_strength, latent_image, noise, sampler, sigmas,
                  callback, x0_output, disable_pbar):
        """
        Sample a latent using CFGGuider with LoRA patches applied.
        Args:
            guider: Input CFGGuider to copy conds/cfg from
            lora_patch_dict: LoRA key dict from merge
            lora_strength: Strength to apply LoRA at
            latent_image: Input latent dict {"samples": (B, 4, lH, lW)}
            noise: Noise provider (NOISE type)
            sampler: ComfyUI sampler
            sigmas: Sigmas schedule
            callback: Per-step callback (drives the shared progress bar + previews)
            x0_output: Dict the callback writes the latest x0 into, for the denoised output
            disable_pbar: Whether to disable the sampler's internal progress bar
        Returns:
            {"samples": Tensor (B, 4, lH, lW)}
        """
        model_patcher = guider.model_patcher

        new_model_patcher = model_patcher.clone()
        new_model_patcher.add_patches(lora_patch_dict, lora_strength)

        # Rebuild the guider around the LoRA-patched model, preserving its
        # exact subclass/state (e.g. Guider_DualModel's separate uncond model).
        new_guider = rebuild_guider_with_patches(guider, new_model_patcher)

        latent = latent_image.copy()
        latent_samples = latent["samples"]  # (B, 4, lH, lW)
        latent_samples = comfy.sample.fix_empty_latent_channels(
            new_model_patcher, latent_samples,
            latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None)
        )
        latent["samples"] = latent_samples

        noise_mask = latent.get("noise_mask", None)

        samples = new_guider.sample(
            noise.generate_noise(latent), latent_samples, sampler, sigmas,
            denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples  # (B, 4, lH, lW) with B=1
        if "x0" in x0_output:
            x0_out = new_model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [s.shape for s in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out
        return out_denoised

    @staticmethod
    def annotate_image(parameter_name: str, parameter_value: float, img_tensor: torch.Tensor,
                      parameter_name_2: str = None, parameter_value_2: float = None) -> torch.Tensor:
        """
        Annotate an image with parameter information.
        Args:
            parameter_name: Name of the first parameter
            parameter_value: Value of the first parameter
            img_tensor: Image tensor, channels-last (H, W, C) or channels-first (C, H, W)
            parameter_name_2: Optional name of the second parameter
            parameter_value_2: Optional value of the second parameter
        Returns:
            Annotated image tensor with title prepended: (title_H + H, W, 3) channels-last
        """
        title_font = load_font()

        # Normalize img_tensor to (H, W, C) channels-last
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(-1).expand(-1, -1, 3)
        elif img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
            img_tensor = img_tensor.permute(1, 2, 0)

        # img_tensor: (H, W, C)
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        if parameter_name_2 is not None and parameter_value_2 is not None:
            title = f"{parameter_name}: {parameter_value:.3f}, {parameter_name_2}: {parameter_value_2:.3f}"
        else:
            title = f"{parameter_name}: {parameter_value:.3f}"

        max_text_width = img.width - 100  # small margin
        lines = []
        for word in title.split():
            if not lines:
                lines.append(word)
            else:
                test_line = lines[-1] + ' ' + word
                test_width = title_font.getbbox(test_line)[2]
                if test_width <= max_text_width:
                    lines[-1] = test_line
                else:
                    lines.append(word)

        # Split any lines that are still too wide (long single words)
        wrapped_lines = []
        for line in lines:
            line_width = title_font.getbbox(line)[2]
            if line_width <= max_text_width:
                wrapped_lines.append(line)
            else:
                current = ""
                for char in line:
                    test = current + char
                    if title_font.getbbox(test)[2] > max_text_width:
                        wrapped_lines.append(current)
                        current = char
                    else:
                        current = test
                if current:
                    wrapped_lines.append(current)
        lines = wrapped_lines

        title_padding = 6
        line_height = title_font.getbbox("A")[3] + title_padding
        title_text_height = line_height * len(lines) + title_padding
        title_text_image = Image.new('RGB', (img.width, title_text_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(title_text_image)
        for i, line in enumerate(lines):
            line_width = title_font.getbbox(line)[2]
            draw.text(
                ((img.width - line_width) // 2, i * line_height + title_padding // 2),
                line,
                font=title_font,
                fill=(255, 255, 255)
            )
        # title_text_image_tensor: (title_H, W, 3) channels-last
        title_text_image_tensor = torch.tensor(np.array(title_text_image).astype(np.float32) / 255.0)

        # (title_H, W, 3) cat (H, W, 3) -> (title_H+H, W, 3)
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
