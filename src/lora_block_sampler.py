import logging

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw, ImageChops

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview

from nodes import VAEDecode

from .comfy_util import load_as_comfy_lora, rebuild_guider_with_patches
from .architectures import sd_lora, dit_lora
from .utility import FONTS_DIR


class LoRABlockSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "lora": ("LoRABundle",),
                "vae": ("VAE",),
                "model": ("MODEL",),
                "bock_sampling_mode": (["round_robin_exclude", "round_robin_include"],),
                "image_display": (["image", "image_diff"],)
            }
        }
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "image_grid")
    FUNCTION = "sample"
    CATEGORY = "LoRA PowerMerge/sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image, lora, vae, model,
               bock_sampling_mode, image_display):
        if 'lora' not in lora or lora['lora'] is None:
            lora['lora'] = load_as_comfy_lora(lora, model)

        action = "include" if bock_sampling_mode == "round_robin_include" else "exclude"

        patch_dict = lora['lora']

        logging.info(f"PM LoRABlockSampler: Sample keys from patch_dict: {list(patch_dict.keys())[:3]}")

        def get_string_key(key):
            if isinstance(key, tuple) and len(key) > 0 and isinstance(key[0], str):
                return key[0]
            elif isinstance(key, str):
                return key
            return None

        string_keys_for_detection = []
        for key in patch_dict.keys():
            str_key = get_string_key(key)
            if str_key:
                string_keys_for_detection.append(str_key)

        detection_dict = {get_string_key(k): v for k, v in patch_dict.items() if get_string_key(k)}
        arch = dit_lora.detect_architecture(detection_dict)
        if arch == "dit":
            logging.info("PM LoRABlockSampler: Detected DiT architecture")
            detect_fn = dit_lora.detect_block_names
        else:
            logging.info("PM LoRABlockSampler: Using SD/SDXL architecture")
            detect_fn = sd_lora.detect_block_names

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

        logging.info(f"PM LoRABlockSampler: Detected main blocks: {main_blocks}")
        main_blocks = ["NONE", "ALL"] + sorted(list(main_blocks))

        latents_out = self.do_sample(guider, patch_dict, lora['strength_model'], latent_image,
                                      noise, sampler, sigmas, detect_fn, action, main_blocks)

        # Repack the output: (num_blocks, 4, lH, lW). Each block latent is (1, 4, lH, lW),
        # so concatenate along the batch axis (stack would add a spurious 5th dim and
        # break vae.decode).
        out = {
            "samples": torch.cat([s['samples'] for s in latents_out], dim=0)
        }

        print(f"PM LoRABlockSampler: VAE info - latent_format: {getattr(vae, 'latent_channels', '?')}ch, first_stage: {type(getattr(vae, 'first_stage_model', None)).__name__}, output_channels: {getattr(vae, 'output_channels', '?')}, upscale_ratio: {getattr(vae, 'upscale_ratio', '?')}")

        grid_images = []
        if vae is not None:
            latent_in = out['samples']
            print(f"PM LoRABlockSampler: latents shape={latent_in.shape}, dtype={latent_in.dtype}, block means: {[f'{latent_in[i].float().mean().item():.3f}' for i in range(latent_in.shape[0])]}")

            raw_decoded = vae.decode(out['samples'])
            logging.info(f"PM LoRABlockSampler: VAE decode returned shape={raw_decoded.shape}, dtype={raw_decoded.dtype}, device={raw_decoded.device}")
            logging.info(f"PM LoRABlockSampler: VAE decode dims: " + str(list(raw_decoded.shape)))

            # Auto-detect channel axis: find which axis has value ~3 (RGB) and which has large values
            for dim_idx in range(raw_decoded.ndim):
                dim_size = raw_decoded.shape[dim_idx]
                if dim_size == 3:
                    ch_mean = raw_decoded.select(dim_idx, 0).float().mean().item()
                    logging.info(f"PM LoRABlockSampler: dim {dim_idx} == 3 (RGB candidates), channel-0 mean={ch_mean:.1f}")

            if raw_decoded.ndim == 5:
                logging.info(f"PM LoRABlockSampler: 5D tensor detected")
                # (B, T, C_or_H, H, W) or (B, C, T, H, W)
                logging.info(f"PM LoRABlockSampler: trying to identify format...")
                # Try: dim0=B, dim1=T, dim2=channels
                if raw_decoded.shape[2] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, T, 3, H, W) - channels at dim2")
                    raw_decoded = raw_decoded.movedim(2, -1)  # (B, T, H, W, 3)
                    raw_decoded = raw_decoded.reshape(-1, raw_decoded.shape[-3], raw_decoded.shape[-2], raw_decoded.shape[-1])
                elif raw_decoded.shape[4] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, T, H, W, 3) - already channels-last")
                elif raw_decoded.shape[3] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, T, H, 3, W) - weird")
            elif raw_decoded.ndim == 4:
                if raw_decoded.shape[3] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, H, W, 3) - channels-last, OK")
                elif raw_decoded.shape[1] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, 3, H, W) - channels-first, converting")
                    raw_decoded = raw_decoded.movedim(1, -1)
                elif raw_decoded.shape[-1] != 3:
                    # Last dim is not 3, try to find channels dim
                    for di in range(raw_decoded.ndim):
                        if raw_decoded.shape[di] == 3:
                            logging.info(f"PM LoRABlockSampler: Channels at dim {di}, converting to channels-last")
                            raw_decoded = raw_decoded.movedim(di, -1)
                            break
            elif raw_decoded.ndim == 3:
                if raw_decoded.shape[0] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (3, H, W) - single image channels-first")
                    raw_decoded = raw_decoded.movedim(0, -1).unsqueeze(0)
                elif raw_decoded.shape[-1] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, H, W, 3) - channels-last")
                elif raw_decoded.shape[-3] == 3:
                    logging.info(f"PM LoRABlockSampler: Format (B, H, 3, W) - needs permute")
                    raw_decoded = raw_decoded.movedim(-3, -1)

            if raw_decoded.ndim == 4:
                if raw_decoded.shape[0] == 9:
                    images = [raw_decoded[i] for i in range(9)]
                else:
                    images = list(raw_decoded)
            else:
                logging.info(f"PM LoRABlockSampler: WARNING - unexpected ndim={raw_decoded.ndim} after conversion, shape={raw_decoded.shape}")
                images = []

            for i, img in enumerate(images):
                if img.ndim == 3 and img.shape[-1] == 3:
                    logging.info(f"PM LoRABlockSampler: image[{i}] {img.shape}: R={img[:,:,0].float().mean().item():.2f} G={img[:,:,1].float().mean().item():.2f} B={img[:,:,2].float().mean().item():.2f} all_in_0_1={img.min().item():.3f}-{img.max().item():.3f}")
                else:
                    logging.info(f"PM LoRABlockSampler: image[{i}] unexpected shape={img.shape}")

            font_path = f"{FONTS_DIR}/ShareTechMono-Regular.ttf"
            try:
                title_font = ImageFont.truetype(font_path, size=48)
            except OSError:
                logging.warning(f"PM LoRABlockSampler: Font not found at {font_path}, using default font.")
                title_font = ImageFont.load_default()

            img_diff_target = None
            for img_tensor, block_name in zip(images, main_blocks):
                # img_tensor: (H, W, C) channels-last
                i = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

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

                # title_text_image_tensor: (title_H, W, 4) -- RGBA
                title_np = np.array(title_text_image)
                title_np = title_np[:, :, :3]  # Drop alpha channel -> (title_H, W, 3)
                title_text_image_tensor = torch.tensor(title_np.astype(np.float32) / 255.0)

                if image_display == "image_diff":
                    if block_name not in ("NONE", "ALL") and img_diff_target is not None:
                        img_pil = ImageChops.difference(img, img_diff_target)
                        img_np = np.array(img_pil).astype(np.float32) / 255.0
                        img_tensor = torch.tensor(img_np)
                    else:
                        img_tensor = img_tensor.cpu()

                # Normalize img_tensor to (H, W, C) channels-last, C=3
                if img_tensor.dim() == 2:
                    img_tensor = img_tensor.unsqueeze(-1).expand(-1, -1, 3)
                elif img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
                    img_tensor = img_tensor.permute(1, 2, 0)

                # (title_H, W, 3) cat (H, W, 3) -> (title_H+H, W, 3)
                out_image = torch.cat([title_text_image_tensor, img_tensor], 0)
                grid_images.append(out_image)

        # Pad all images to the same height so torch.stack succeeds
        if grid_images:
            max_h = max(img.shape[0] for img in grid_images)
            max_w = grid_images[0].shape[1]
            num_chans = grid_images[0].shape[2]
            padded = []
            for img in grid_images:
                h = img.shape[0]
                if h < max_h:
                    pad = torch.zeros(max_h - h, max_w, num_chans, dtype=img.dtype, device=img.device)
                    img = torch.cat([img, pad], dim=0)
                padded.append(img)
            grid_images = padded

        # grid_images: list of (max_H, W, 3) — all same height
        return out, torch.stack(grid_images),

    @staticmethod
    def do_sample(guider, patch_dict, lora_strength, latent_image, noise, sampler, sigmas,
                  detect_fn, action, main_blocks):
        latents_out = []

        model_patcher = guider.model_patcher

        logging.info(f"PM LoRABlockSampler: guider.original_conds keys = {list(guider.original_conds.keys())}")
        logging.info(f"PM LoRABlockSampler: guider.conds keys = {list(getattr(guider, 'conds', {}).keys())}")
        for k in guider.original_conds:
            v = guider.original_conds[k]
            logging.info(f"PM LoRABlockSampler: original_conds['{k}'] = list of {len(v)} items")

        steps = sigmas.shape[-1] - 1
        num_blocks = len(main_blocks)
        x0_per_block = {}

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Build a previewer (TAESD / latent2rgb) so previews appear live during sampling,
        # mirroring latent_preview.prepare_callback but with a progress bar spanning all blocks.
        previewer = latent_preview.get_previewer(
            model_patcher.load_device, model_patcher.model.latent_format
        )
        pbar = comfy.utils.ProgressBar(num_blocks * steps) if not disable_pbar else None

        def make_callback(block_idx):
            b_idx = block_idx
            block_start = b_idx * steps

            def cb(step, x0, x, total_steps):
                x0_per_block[b_idx] = x0
                if pbar is not None:
                    preview_bytes = None
                    if previewer is not None:
                        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    pbar.update_absolute(block_start + step + 1, num_blocks * steps, preview_bytes)

            return cb

        for block_idx, block in enumerate(main_blocks):
            patch_dict_filtered = {}
            for orig_key, value in patch_dict.items():
                str_key = orig_key if isinstance(orig_key, str) else (orig_key[0] if isinstance(orig_key, tuple) and len(orig_key) > 0 else None)
                if not str_key:
                    continue
                if block == "NONE":
                    continue
                if block == "ALL":
                    patch_dict_filtered[orig_key] = value
                else:
                    block_names = detect_fn(str_key)
                    key_main_block = block_names["main_block"] if block_names and "main_block" in block_names else None
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

            new_model_patcher = model_patcher.clone()
            new_model_patcher.add_patches(patch_dict_filtered, lora_strength)

            # Rebuild the guider around the LoRA-patched model, preserving its
            # exact subclass/state (e.g. Guider_DualModel's separate uncond model).
            new_guider = rebuild_guider_with_patches(guider, new_model_patcher)
            logging.info(f"PM LoRABlockSampler: new_guider type = {type(new_guider).__name__}, "
                         f"conds keys = {list(new_guider.conds.keys())}, "
                         f"uncond_model = {getattr(new_guider, 'uncond_model_patcher', None) is not None}, "
                         f"cfg = {getattr(new_guider, 'cfg', None)}")

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
                denoise_mask=noise_mask, callback=make_callback(block_idx), disable_pbar=disable_pbar, seed=noise.seed
            )
            logging.info(f"PM LoRABlockSampler: Block {block_idx} ({block}) samples shape={samples.shape}, is_nested={getattr(samples, 'is_nested', False)}, dtype={samples.dtype}, device={samples.device}, mean={samples.float().mean().item():.4f}")
            samples = samples.to(comfy.model_management.intermediate_device())

            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples  # (B, 4, lH, lW) with B=1
            block_x0 = x0_per_block.get(block_idx)
            if block_x0 is not None:
                # Scale x0 from the model's internal latent space into the standard
                # latent format the VAE expects. Without this the decode looks like a
                # washed-out depth/diff map.
                x0_out = new_model_patcher.model.process_latent_out(block_x0.cpu())
                if getattr(samples, 'is_nested', False):
                    latent_shapes = [s.shape for s in samples.unbind()]
                    x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
                out_denoised = latent.copy()
                out_denoised["samples"] = x0_out
                logging.info(f"PM LoRABlockSampler: Block {block_idx} ({block}) USING x0 (denoised). x0 mean={x0_out.float().mean().item():.4f}")
            else:
                out_denoised = out
                logging.info(f"PM LoRABlockSampler: Block {block_idx} ({block}) USING samples (no x0). mean={samples.float().mean().item():.4f}")
            latents_out.append(out_denoised)

        return latents_out