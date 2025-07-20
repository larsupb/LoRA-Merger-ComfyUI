from typing import Dict, Optional, Any, List

import torch
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ImmutableMap
from mergekit.io.tasks import GatherTensors
from mergekit.sparsify import SparsificationMethod, RescaleNorm, sparsify

from comfy.utils import ProgressBar  # Assuming this is thread-safe or replaced


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
