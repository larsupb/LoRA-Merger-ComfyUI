import logging
import os

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw, ImageChops

from nodes import VAEDecode
from comfy_extras.nodes_custom_sampler import SamplerCustom

from .comfy_util import load_as_comfy_lora
from .architectures import sd_lora, dit_lora

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")


class LoRABlockSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
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
                "lora": ("LoRABundle",),
                "vae": ("VAE",),
                "bock_sampling_mode": (["round_robin_exclude", "round_robin_include"],),
                "image_display": (["image", "image_diff"],)
            }
        }
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "image_grid")
    FUNCTION = "sample"
    CATEGORY = "LoRA PowerMerge/sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas,
               latent_image, lora, vae, bock_sampling_mode, image_display):
        if 'lora' not in lora or lora['lora'] is None:
            lora['lora'] = load_as_comfy_lora(lora, model)

        action = "include" if bock_sampling_mode == "round_robin_include" else "exclude"

        patch_dict = lora['lora']

        # Debug: Log first few keys to understand structure
        sample_keys = list(patch_dict.keys())[:3]
        logging.info(f"PM LoRABlockSampler: Sample keys from patch_dict: {sample_keys}")

        # Helper function to extract string key from various formats
        def get_string_key(key):
            """Extract string key from tuple or return as-is if already string"""
            if isinstance(key, tuple) and len(key) > 0 and isinstance(key[0], str):
                return key[0]
            elif isinstance(key, str):
                return key
            return None

        # Build a mapping of string keys for detection only
        # Keep original keys intact for patch application
        string_keys_for_detection = []
        for key in patch_dict.keys():
            str_key = get_string_key(key)
            if str_key:
                string_keys_for_detection.append(str_key)

        # Auto-detect architecture using string keys
        detection_dict = {get_string_key(k): v for k, v in patch_dict.items() if get_string_key(k)}
        arch = dit_lora.detect_architecture(detection_dict)
        if arch == "dit":
            logging.info("PM LoRABlockSampler: Detected DiT architecture")
            detect_fn = dit_lora.detect_block_names
        else:
            logging.info("PM LoRABlockSampler: Using SD/SDXL architecture")
            detect_fn = sd_lora.detect_block_names

        # Detect main blocks using string keys
        main_blocks = set()
        for key in patch_dict.keys():
            str_key = get_string_key(key)
            if not str_key:
                logging.warning(f"PM LoRABlockSampler: Skipping unsupported key type: {type(key)} = {key}")
                continue
            block_names = detect_fn(str_key)
            if block_names is None:
                continue
            main_blocks.add(block_names["main_block"])

        out = []
        kSampler = SamplerCustom()

        logging.info(f"PM LoRABlockSampler: Detected main blocks: {main_blocks}")
        main_blocks = ["NONE", "ALL"] + sorted(list(main_blocks))
        for block in main_blocks:
            patch_dict_filtered = {}
            for orig_key, value in patch_dict.items():
                # Get string representation for filtering logic
                str_key = get_string_key(orig_key)
                if not str_key:
                    continue
                if block == "NONE":
                    continue
                if block == "ALL":
                    # Use original key to preserve metadata
                    patch_dict_filtered[orig_key] = value
                else:
                    # Detect which main_block this key belongs to
                    block_names = detect_fn(str_key)
                    key_main_block = block_names["main_block"] if block_names and "main_block" in block_names else None

                    # Filter based on detected main_block
                    if key_main_block:
                        if (action == "include" and key_main_block == block) or \
                           (action == "exclude" and key_main_block != block):
                            patch_dict_filtered[orig_key] = value

            if block == "NONE":
                logging.info("PM LoRABlockSampler: Do not apply any of the patches.")
            elif block == "ALL":
                logging.info(f"PM LoRABlockSampler: Apply all patches. Total patches: {len(patch_dict.keys())}")
            else:
                logging.info(f"PM LoRABlockSampler: {action} block {block} from sampling, "
                             f"remaining patches: {len(patch_dict_filtered)}")

            new_model_patcher = model.clone()
            new_model_patcher.add_patches(patch_dict_filtered, lora['strength_model'])

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
