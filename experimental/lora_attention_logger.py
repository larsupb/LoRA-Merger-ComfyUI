import io
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt

import comfy
from ..architectures.sd_lora import detect_block_names

# Create a global variable to store the norm values for each layer
# Format: { layer_name: [norm1, norm2, ...] }
layer_step_log = dict()

'''
Patch the KSampler to inject our step callback
This allows us to log the step index during sampling.
'''
original_sample = comfy.samplers.KSampler.sample


def patched_sample(self, *args, **kwargs):
    # extra = kwargs.get("extra_options", {})
    # if "callback" not in extra:
    # extra["callback"] = step_callback
    # kwargs["extra_options"] = extra
    return original_sample(self, *args, **kwargs)


comfy.samplers.KSampler.sample = patched_sample


def make_logging_hook(layer_name):
    def hook_fn(module, input, output):
        with torch.no_grad():
            # step_index = get_step_index_fn()
            norm = output.norm(dim=-1).mean().item()
            if layer_name not in layer_step_log:
                layer_step_log[layer_name] = []
            layer_step_log[layer_name].append(norm)

    return hook_fn


def register_attention_hooks(model):
    for name, module in model.named_modules():
        # Customize this filter as needed for LoRA models
        if "attn" in name.lower() or "cross_attn" in name.lower():
            try:
                module.register_forward_hook(make_logging_hook(name))
            except Exception as e:
                print(f"Failed to register hook on {name}: {e}")
    return model


class LoRAAttentionLogger:
    """
    Wraps a model (UNet or any LoRA-modified model) and installs hooks
    that log attention activity during the forward pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_hooks"
    CATEGORY = "LoRA PowerMerge/Analytics"

    def apply_hooks(self, model):
        # The UNet is inside model.model.diffusion_model if coming from base nodes
        target_model = getattr(model, "model", model)

        print("[LoRAAttentionLogger] Registering attention hooks...")
        register_attention_hooks(target_model)

        global layer_step_log
        layer_step_log = defaultdict(lambda: defaultdict(list))

        return (model,)


class LoRAAttentionPlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("activity_plot",)
    FUNCTION = "plot_activity"
    CATEGORY = "custom"

    def plot_activity(self, latent):
        if not layer_step_log:
            raise ValueError("No attention activity data found.")

        # Group the layers by their block names
        layer_data = []
        for layer_name, norms in layer_step_log.items():
            data0 = [layer_name, norms] + list(detect_block_names(layer_name).values())
            layer_data.append(data0)
        df = pd.DataFrame(layer_data, columns=["layer_name", "norms",
                                               "block_type", "block_idx", "inner_idx", "component",
                                               "main_block", "sub_block", "transformer_idx"])

        # Convert to image
        pil_img = self.generate_plot_image(df)
        pil_detail_images: List[Image] = self.generate_block_detail_images(df)

        image_tensors = self.pil_to_tensor_batch([pil_img] + pil_detail_images)

        # Clear data after processing
        layer_step_log.clear()  # Clear the log after processing

        return (image_tensors,)

    def generate_plot_image(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        subplot_map = {
            "input_blocks": axes[0, 0],
            "middle_block": axes[0, 1],
            "output_blocks": axes[1, 0],
        }

        subplot_titles = {
            "input_blocks": "Input Blocks",
            "middle_block": "Middle Blocks",
            "output_blocks": "Output Blocks",
        }

        # Track which subplot had content
        filled_axes = set()

        # Determine number of steps from the first row
        step_count = len(df["norms"].iloc[0])
        x_values = np.arange(step_count)

        # Group rows by block_type
        for block_type, ax in subplot_map.items():
            block_df = df[df["block_type"] == block_type]

            if block_df.empty:
                ax.axis('off')
                continue

            filled_axes.add(block_type)

            # Group by main_block (e.g., input1, middle2...) for plotting
            for block_name, group in block_df.groupby("main_block"):
                norms_array = np.stack(group["norms"].values)  # shape: (num_layers, step_count)
                summed_norms = norms_array.sum(axis=0)  # shape: (step_count,)

                ax.plot(x_values, summed_norms, label=block_name, linewidth=1)

            ax.set_title(subplot_titles[block_type])
            ax.set_xlabel("Denoising Step")
            ax.set_ylabel("Output Norm")
            ax.legend(fontsize='x-small', loc='upper right')

        # Hide unused subplot
        axes[1, 1].axis('off')

        plt.tight_layout()

        # Save to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        buf.close()
        plt.close()

        return image

    def generate_block_detail_images(self, df: pd.DataFrame) -> List[Image.Image]:
        images = []
        components = ["attn1", "attn2", "ff"]
        component_titles = {
            "attn1": "Attention",
            "attn2": "Cross-Attention",
            "ff": "Feedforward",
        }

        step_count = len(df["norms"].iloc[0])
        x_values = np.arange(step_count)

        for main_block, group in df.groupby("main_block"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            flat_axes = axes.flatten()

            for i, component in enumerate(components):
                ax = flat_axes[i]
                comp_df = group[group["component"] == component]

                if comp_df.empty:
                    ax.axis('off')
                    continue

                # Plot series grouped by transformer_idx (or "others")
                for transformer_idx, sub_df in comp_df.groupby(
                        comp_df["transformer_idx"].apply(lambda x: x if pd.notna(x) else "others")
                ):
                    norms_array = np.stack(sub_df["norms"].values)  # (num_layers, step_count)
                    summed_norms = norms_array.sum(axis=0)
                    label = f"{component_titles[component]} - {transformer_idx}"
                    ax.plot(x_values, summed_norms, label=label, linewidth=1)

                ax.set_title(component_titles[component])
                ax.set_xlabel("Denoising Step")
                ax.set_ylabel("Output Norm")
                ax.legend(fontsize="x-small", loc="upper right")

            # Leave last subplot empty
            flat_axes[3].axis('off')

            plt.suptitle(f"Block: {main_block}", fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            buf.close()
            plt.close()

            images.append(image)

        return images

    def pil_to_tensor(self, image: Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=0)  # Shape: (1, H, W, 3)
        return torch.from_numpy(image_np)

    def pil_to_tensor_batch(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in images:
            tensor = self.pil_to_tensor(img)
            tensor = tensor.squeeze(0)  # (H, W, 3)
            tensors.append(tensor)  # Convert each image to tensor
        stack = torch.stack(tensors)
        print('Shape of stacked tensors:', stack.shape)  # Debugging output
        return stack
