import logging
import os

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw, ImageChops

from nodes import VAEDecode
from .utility import load_as_comfy_lora
from .architectures.sd_lora import detect_block_names
from comfy_extras.nodes_custom_sampler import SamplerCustom

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")


class LoRABlockSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": (
                    "INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}
                ),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "positive_text": ("STRING",),
                "negative_text": ("STRING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "lora": ("LoRA",),
                "vae": ("VAE",),
                "bock_sampling_mode": (["round_robin_exclude", "round_robin_include"],),
                "image_display": (["image", "image_diff"],)
            }
        }
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "image_grid")
    FUNCTION = "sample"
    CATEGORY = "LoRA PowerMerge/sampling"

    def sample(self, model, clip, add_noise, noise_seed, cfg, positive_text, negative_text, sampler, sigmas,
               latent_image, lora, vae, bock_sampling_mode, image_display):
        if lora["lora"] is None:
            lora['lora'] = load_as_comfy_lora(lora, model, clip)

        action = "include" if bock_sampling_mode == "round_robin_include" else "exclude"

        patch_dict = lora['lora']
        main_blocks = set()
        for layer_key, value in patch_dict.items():
            block_names = detect_block_names(layer_key)
            if block_names is None:
                continue
            main_blocks.add(block_names["main_block"])

        out = []
        kSampler = SamplerCustom()

        logging.info(f"PM LoRABlockSampler: Detected main blocks: {main_blocks}")
        main_blocks = ["NONE", "ALL"] + sorted(list(main_blocks))
        for block in main_blocks:
            patch_dict_filtered = {}
            for layer_key, value in patch_dict.items():
                if block == "NONE":
                    continue
                if block == "ALL":
                    patch_dict_filtered[layer_key] = value
                else:
                    if (action == "include" and block in layer_key) or (action == "exclude" and block not in layer_key):
                        patch_dict_filtered[layer_key] = value

            if block == "NONE":
                logging.info("PM LoRABlockSampler: Do not apply any of the patches.")
            elif block == "ALL":
                logging.info(f"PM LoRABlockSampler: Apply all patches. Total patches: {len(patch_dict.keys())}")
            else:
                logging.info(f"PM LoRABlockSampler: {action} block {block} from sampling, "
                             f"remaining patches: {len(patch_dict_filtered)}")

            new_model_patcher = model.clone()
            new_model_patcher.add_patches(patch_dict_filtered, lora['strength_model'])
            new_clip = clip.clone()
            new_clip.add_patches(patch_dict_filtered, lora['strength_clip'])

            # Use patched clip to condition the model
            tokens = new_clip.tokenize(positive_text)
            positive = new_clip.encode_from_tokens_scheduled(tokens)

            tokens = new_clip.tokenize(negative_text)
            negative = new_clip.encode_from_tokens_scheduled(tokens)

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
            out.append(denoised)

        # Repack the output
        out = {
            "samples": torch.stack([s['samples'].squeeze(0) for s in out])
        }

        grid_images = []
        if vae is not None:
            vae_decoder = VAEDecode()
            images = list(vae_decoder.decode(vae, out)[0])

            # Load a font
            font_path = f"{FONTS_DIR}/ShareTechMono-Regular.ttf"
            try:
                title_font = ImageFont.truetype(font_path, size=48)
            except OSError:
                logging.warning(f"PM LoRABlockSampler: Font not found at {font_path}, using default font.")
                title_font = ImageFont.load_default()

            img_diff_target = None
            for img_tensor, block_name in zip(images, main_blocks):
                i = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Write a white text on the image indicating the block name
                if block_name == "NONE":
                    title = "No LoRA blocks applied"
                    if action == "include":
                        img_diff_target = img
                elif block_name == "ALL":
                    title = "All LoRA blocks applied"
                    if action == "exclude":
                        img_diff_target = img
                else:
                    title = f"{action.capitalize()} block: {block_name}"
                title_width = title_font.getbbox(title)[2]
                title_padding = 6
                title_line_height = (title_font.getmask(title).getbbox()[3] + title_font.getmetrics()[1] +
                                     title_padding * 2)
                title_text_height = title_line_height
                title_text_image = Image.new('RGB', (img.width, title_text_height), color=(0, 0, 0, 0))

                draw = ImageDraw.Draw(title_text_image)
                draw.text((img.width // 2 - title_width // 2, title_padding), title, font=title_font,
                          fill=(255, 255, 255))
                # Convert the title text image to a tensor
                title_text_image_tensor = torch.tensor(np.array(title_text_image).astype(np.float32) / 255.0)

                if image_display == "image_diff":
                    # Calculate the difference
                    if block_name not in ("NONE", "ALL") and img_diff_target is not None:
                        img_tensor = ImageChops.difference(img, img_diff_target)
                        # Convert the image difference to a tensor
                        img_tensor = torch.tensor(np.array(img_tensor).astype(np.float32) / 255.0)

                out_image = torch.cat([title_text_image_tensor, img_tensor], 0)
                grid_images.append(out_image)

        grid_images = torch.stack(grid_images)
        return out, grid_images,
