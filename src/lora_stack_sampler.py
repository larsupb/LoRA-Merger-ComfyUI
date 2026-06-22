import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy.sd import VAE
from .architectures import LORA_STACK, LORA_WEIGHTS
from .comfy_util import rebuild_guider_with_patches
from .utility import load_font


class LoRAStackSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "lora_key_dicts": ("LoRAStack", {"tooltip": "The dictionary containing LoRA names and key weights."}),
                "lora_strengths": ("LoRAWeights", {"tooltip": "The LoRA weighting to apply."}),
            }
        }
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("latents", "images", "image_grid")
    FUNCTION = "sample"
    CATEGORY = "LoRA PowerMerge/sampling"
    DESCRIPTION = "Samples images by iterating over the given LoRA key dictionary and applying the LoRA weights."

    def sample(self, vae: VAE, noise, guider, sampler, sigmas,
               latent_image, lora_key_dicts: LORA_STACK = None, lora_strengths: LORA_WEIGHTS = None):
        if lora_key_dicts is None or lora_strengths is None:
            raise ValueError("key_dicts and lora_weighting must be provided.")

        # latents_out: list of {"samples": Tensor (B, 4, lH, lW), ...}
        latents_out = self.do_sample(guider, lora_key_dicts, lora_strengths, latent_image, noise, sampler, sigmas)

        # Create a grid of images with LoRA names and strengths
        names = list(lora_key_dicts.keys())
        weights = list(lora_strengths.values())
        grid_single_images, image_grid = self.image_grid(names, weights, [s['samples'] for s in latents_out], vae)

        # Repack the output: Tensor (N, 4, lH, lW) where N = num_loras
        latents_out = {
            "samples": torch.cat([s['samples'] for s in latents_out], dim=0)
        }
        return latents_out, grid_single_images, image_grid

    @staticmethod
    def image_grid(names, strengths, batch_latents, vae) -> tuple[Tensor, Tensor]:
        """
        Create an image grid with batches on Y-axis and LoRA names on X-axis.
        Args:
            names: List of LoRA names.
            strengths: List of strengths corresponding to each LoRA name.
            batch_latents: List of latents, where each tensor is a batch of images.
            vae: The VAE model used for decoding the latent.
        Returns:
            Tuple of (all images tensor, grid image tensor)
            - all images:  Tensor (N, H, W, C), N = num_loras * num_batches
            - grid image:  Tensor (1, num_batches*H, num_loras*W, C)
        """
        # batch_latents: list of Tensor (B, 4, lH, lW), one per LoRA
        # grid_images indexed as [batch * num_loras + lora]
        # so batch items for the same LoRA are at indices: [lora_idx, lora_idx + num_loras, ...]
        num_loras = len(names)
        grid_images = [[] for _ in range(num_loras)]

        for lora_idx, (n, w, l) in enumerate(zip(names, strengths, batch_latents)):
            n = n.split('/')[-1].split('.')[0]

            # vae.decode returns: Tensor (B, H, W, C) channels-last
            images = vae.decode(l)  # (B, H, W, C)
            if len(images.shape) == 5:
                # Video VAE: (B, T, C, H, W) -> (B*T, H, W, C)
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

            # Ensure (B, H, W, C) even when B=1
            images = images.squeeze(dim=0)  # (H, W, C) when B=1, (B, H, W, C) when B>1
            if images.dim() == 3:
                images = images.unsqueeze(0)  # (1, H, W, C)
            # images is now always (B, H, W, C)

            for img in images:
                # img: (H, W, C)
                out_image = LoRAStackSampler.annotate_image(n, w, img)
                grid_images[lora_idx].append(out_image)

        # grid_images: list of lists, grid_images[lora_idx][batch_idx] = (H, W, C)
        num_batches = len(grid_images[0]) if grid_images and grid_images[0] else 0

        if not grid_images or not grid_images[0]:
            return torch.tensor([]), torch.tensor([])

        all_images = [img for row in grid_images for img in row]
        max_h = max(img.shape[0] for img in all_images)
        max_w = all_images[0].shape[1]
        num_img_chans = all_images[0].shape[2]

        # Layout: batches on Y (rows), LoRAs on X (columns)
        # img_grid: (1, num_batches*H, num_loras*W, C)
        img_grid = torch.zeros(1, num_batches * max_h, num_loras * max_w, num_img_chans)
        for lora_idx in range(num_loras):
            for batch_idx in range(num_batches):
                img_grid[:, batch_idx * max_h:(batch_idx + 1) * max_h,
                        lora_idx * max_w:(lora_idx + 1) * max_w] = grid_images[lora_idx][batch_idx]

        # Flatten annotated images: (num_loras, num_batches, H, W, C) -> (num_loras*num_batches, H, W, C)
        # Normalize all images to the same height so torch.stack succeeds
        normalized = []
        for img in all_images:
            h = img.shape[0]
            if h < max_h:
                pad = torch.zeros(max_h - h, max_w, num_img_chans, dtype=img.dtype, device=img.device)
                img = torch.cat([img, pad], dim=0)
            normalized.append(img)
        return torch.stack(normalized), img_grid

    @staticmethod
    def annotate_image(name, weighting, img_tensor):
        """
        Annotate an image with LoRA name and strength.
        Args:
            img_tensor: (H, W, C) channels-last, C=3 for RGB
        Returns:
            (title_H + H, W, C) channels-last
        """
        title_font = load_font()

        # img_tensor: (H, W, C)
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        title = f"{name}\nStrength: {weighting['strength_model']:.2f}"
        max_text_width = img.width - 100  # small margin
        char_width = title_font.getbbox("M")[2]
        # Wrap the text, including splitting long single words that exceed max_text_width
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

        # Split any lines that are still too wide (e.g. a single long word exceeds max)
        wrapped_lines = []
        for line in lines:
            line_width = title_font.getbbox(line)[2]
            if line_width <= max_text_width:
                wrapped_lines.append(line)
            else:
                # Character-level wrapping: accumulate chars until width exceeded
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

        # Normalize img_tensor to (H, W, C) channels-last
        # Handles (H, W) grayscale, (C, H, W) channels-first from some VAEs
        if img_tensor.dim() == 2:
            # (H, W) -> (H, W, 1) -> (H, W, 3)
            img_tensor = img_tensor.unsqueeze(-1).expand(-1, -1, 3)
        elif img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
            # (C, H, W) channels-first -> (H, W, C) channels-last
            img_tensor = img_tensor.permute(1, 2, 0)
        # img_tensor is now always (H, W, 3) channels-last

        # Concatenate title above image: (title_H, W, 3) cat (H, W, 3) -> (title_H+H, W, 3)
        out_image = torch.cat([title_text_image_tensor, img_tensor], 0)
        return out_image

    @staticmethod
    def do_sample(guider, key_dicts, lora_strengths, latent_image, noise, sampler, sigmas):
        latents_out = []

        model_patcher = guider.model_patcher

        steps = sigmas.shape[-1] - 1
        num_iters = len(key_dicts)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Build a previewer (TAESD / latent2rgb) so previews appear live during sampling,
        # mirroring latent_preview.prepare_callback but with a single progress bar spanning
        # all LoRA iterations instead of resetting per LoRA.
        previewer = latent_preview.get_previewer(
            model_patcher.load_device, model_patcher.model.latent_format
        )
        pbar = comfy.utils.ProgressBar(num_iters * steps) if not disable_pbar else None

        x0_output = {}

        def make_callback(iter_idx):
            i_idx = iter_idx
            iter_start = i_idx * steps

            def cb(step, x0, x, total_steps):
                x0_output["x0"] = x0
                if pbar is not None:
                    preview_bytes = None
                    if previewer is not None:
                        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    pbar.update_absolute(iter_start + step + 1, num_iters * steps, preview_bytes)

            return cb

        for iter_idx, (lora_name, patch_dict) in enumerate(key_dicts.items()):
            strengths = lora_strengths[lora_name]
            logging.info(f"PM LoRAStackSampler: Applying LoRA {lora_name} with weights {strengths}")

            new_model_patcher = model_patcher.clone()
            new_model_patcher.add_patches(patch_dict, strengths['strength_model'])

            # Rebuild the guider around the LoRA-patched model, preserving its
            # exact subclass/state (e.g. Guider_DualModel's separate uncond model).
            new_guider = rebuild_guider_with_patches(guider, new_model_patcher)

            latent = latent_image.copy()
            latent_image_samples = latent["samples"]  # (B, 4, lH, lW)
            latent_image_samples = comfy.sample.fix_empty_latent_channels(
                new_model_patcher, latent_image_samples,
                latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None)
            )
            latent["samples"] = latent_image_samples

            noise_mask = latent.get("noise_mask", None)

            # noise.generate_noise returns: Tensor (B, 4, lH, lW)
            # new_guider.sample returns: Tensor (B, 4, lH, lW)
            samples = new_guider.sample(
                noise.generate_noise(latent), latent_image_samples, sampler, sigmas,
                denoise_mask=noise_mask, callback=make_callback(iter_idx), disable_pbar=disable_pbar, seed=noise.seed
            )
            samples = samples.to(comfy.model_management.intermediate_device())

            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples  # (B, 4, lH, lW)
            if "x0" in x0_output:
                x0_out = new_model_patcher.model.process_latent_out(x0_output["x0"].cpu())
                if samples.is_nested:
                    latent_shapes = [s.shape for s in samples.unbind()]
                    x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
                out_denoised = latent.copy()
                out_denoised["samples"] = x0_out
            else:
                out_denoised = out
            latents_out.append(out_denoised)

        return latents_out