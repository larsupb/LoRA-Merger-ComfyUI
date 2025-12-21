import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from comfy.sd import VAE
from comfy_extras.nodes_custom_sampler import SamplerCustom
from .architectures import LORA_STACK, LORA_WEIGHTS
from .utility import load_font


class LoRAStackSampler:
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
                "lora_key_dicts": ("LoRAStack", {"tooltip": "The dictionary containing LoRA names and key weights."}),
                "lora_strengths": ("LoRAWeights", {"tooltip": "The LoRA weighting to apply."}),
            }
        }
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("latents", "images", "image_grid")
    FUNCTION = "sample"
    CATEGORY = "LoRA PowerMerge/sampling"
    DESCRIPTION = "Samples images by iterating over the given LoRA key dictionary and applying the LoRA weights."

    def sample(self, model, vae: VAE, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas,
               latent_image, lora_key_dicts: LORA_STACK = None, lora_strengths: LORA_WEIGHTS = None):
        if lora_key_dicts is None or lora_strengths is None:
            raise ValueError("key_dicts and lora_weighting must be provided.")

        latents_out = self.do_sample(add_noise, cfg, lora_key_dicts, lora_strengths, latent_image, model, noise_seed,
                                     positive, negative, sampler, sigmas)

        # Create a grid of images with LoRA names and strengths
        names = list(lora_key_dicts.keys())
        weights = list(lora_strengths.values())
        grid_single_images, image_grid = self.image_grid(names, weights, [s['samples'] for s in latents_out], vae)

        # Repack the output
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
            vae: The VAE model used for decoding the latents.
        Returns:
            Tuple of (all images tensor, grid image tensor)
        """
        grid_images = []

        for n, w, l in zip(names, strengths, batch_latents):
            n = n.split('/')[-1].split('.')[0]

            images = vae.decode(l)
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

            for img in images.squeeze(dim=0):
                out_image = LoRAStackSampler.annotate_image(n, w, img)
                grid_images.append(out_image)

        num_loras = len(names)
        num_batches = len(grid_images) // num_loras

        if not grid_images:
            return torch.tensor([]), torch.tensor([])

        first_image = grid_images[0]

        img_height = first_image.shape[0]
        img_width = first_image.shape[1]
        num_img_chans = first_image.shape[2]

        img_grid = torch.zeros(1, num_batches * img_height, num_loras * img_width, num_img_chans)
        if len(grid_images) > 0:
            for i in range(num_loras):
                for j in range(num_batches):
                    idx = i * num_batches + j
                    img_grid[:, j * img_height:(j + 1) * img_height,
                    i * img_width:(i + 1) * img_width] = grid_images[idx]
        return torch.stack(grid_images), img_grid

    @staticmethod
    def annotate_image(name, weighting, img_tensor):
        title_font = load_font()

        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        title = f"{name}\nStrength: {weighting['strength_model']:.2f}"
        max_text_width = img.width - 100  # small margin
        # Wrap the text
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
        title_text_image_tensor = torch.tensor(np.array(title_text_image).astype(np.float32) / 255.0)
        out_image = torch.cat([title_text_image_tensor, img_tensor], 0)
        return out_image

    @staticmethod
    def do_sample(add_noise, cfg, key_dicts, lora_strengths, latent_image, model, noise_seed, positive,
                  negative, sampler, sigmas):
        latents_out = []

        kSampler = SamplerCustom()
        for lora_name, patch_dict in key_dicts.items():
            strengths = lora_strengths[lora_name]
            logging.info(f"PM LoRAStackSampler: Applying LoRA {lora_name} with weights {strengths}")

            new_model_patcher = model.clone()
            new_model_patcher.add_patches(patch_dict, strengths['strength_model'])

            denoised, _ = kSampler.sample(
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
        return latents_out
