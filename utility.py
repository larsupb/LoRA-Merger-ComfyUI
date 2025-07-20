import logging
import os
from typing import Tuple, Dict

import torch
from PIL import ImageFont

import comfy
from .sd_lora import UP_DOWN_ALPHA_TUPLE

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")

def map_device(device, dtype):
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return device, dtype


def find_network_dim(lora_sd: dict):
    network_dim = None
    for key, value in lora_sd.items():
        if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
            network_dim = value.size()[0]
    return network_dim


def load_as_comfy_lora(lora: dict, model, clip):
    if 'lora_raw' not in lora or lora['lora_raw'] is None:
        raise ValueError("LoRA data is missing. Please provide a valid LoRA dictionary with 'lora_raw' key.")
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    patch_dict = comfy.lora.load_lora(lora['lora_raw'], key_map)
    return patch_dict


def adjust_tensor_dims(ups_downs_alphas: Dict[str, UP_DOWN_ALPHA_TUPLE], apply_svd=False, svd_rank=-1) -> Dict[
    str, UP_DOWN_ALPHA_TUPLE]:
    """
    Checks if tensor dimensions and eventually aligns them to the first tensor with SVD
    Args:
        ups_downs_alphas (list): List of tuples containing up, down tensors and alpha values.
        apply_svd (bool): Whether to apply SVD for dimension adjustment.
        svd_rank (int): Rank for SVD adjustment. If -1, uses the rank of the first tensor.
    """
    up_0, up_1, alpha_0 = next(iter(ups_downs_alphas.values()))
    target_rank = up_0.shape[1] if svd_rank == -1 else svd_rank

    out = {}
    for lora_name, (up, down, alpha) in ups_downs_alphas.items():
        if up.shape[1] != target_rank:
            if not apply_svd:
                raise ValueError(f"LoRA up tensors have different shapes: {up.shape} vs {up_0.shape}. "
                                 f"Turn on apply_svd to True to resize them.")
            original_dtype = up.dtype
            down, up = resize_lora_rank(
                down.to(dtype=torch.float32),
                up.to(dtype=torch.float32),
                target_rank)
            down = down.to(device="cpu", dtype=original_dtype)
            up = up.to(device="cpu", dtype=original_dtype)
        out[lora_name] = (up, down, alpha)
    return out


def resize_lora_rank(down: torch.Tensor, up: torch.Tensor, new_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize LoRA weights to a new rank using SVD.
    This function computes the full LoRA matrix from the low-rank matrices,
    performs Singular Value Decomposition (SVD), and then truncates or pads
    the resulting matrices to the specified new rank.
    Args:
        down (torch.Tensor): The low-rank down matrix.
        up (torch.Tensor): The low-rank up matrix.
        new_dim (int): The desired rank for the LoRA weights.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The resized low-rank matrices.
    """
    # Compute the full LoRA matrix
    try:
        if down.dim() > 2:
            down_ = down.squeeze(-2, -1)  # Ensure down is 2D
        else:
            down_ = down
        if up.dim() > 2:
            up_ = up.squeeze(-2, -1)
        else:
            up_ = up
        W = up_ @ down_  # Shape: (out_features, in_features)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to compute full LoRA matrix: {e}. "
                           "Ensure that 'up' and 'down' tensors are compatible for matrix multiplication.")

    # Perform SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    #U, S, Vh = fast_svd_cupy(W, rank=new_dim)

    # Truncate or pad to new_dim
    rank = min(new_dim, S.size(0))
    U_new = U[:, :rank]  # (out_features, rank)
    S_new = S[:rank]  # (rank,)
    Vh_new = Vh[:rank, :]  # (rank, in_features)

    # Compose new LoRA weights
    S_sqrt = torch.sqrt(S_new)
    up_new = U_new @ torch.diag(S_sqrt)  # (out_features, rank)
    down_new = torch.diag(S_sqrt) @ Vh_new  # (rank, in_features)

    # If padding is needed (i.e., new_dim > rank), zero pad
    if new_dim > rank:
        pad_up = torch.zeros((up.shape[0], new_dim - rank), dtype=up.dtype, device=up.device)
        pad_down = torch.zeros((new_dim - rank, down.shape[1]), dtype=down.dtype, device=down.device)
        up_new = torch.cat([up_new, pad_up], dim=1)
        down_new = torch.cat([down_new, pad_down], dim=0)

    # unsqueeze if necessary
    while down.dim() > down_new.dim():
        down_new = down_new.unsqueeze(-1)
    while up.dim() > up_new.dim():
        up_new = up_new.unsqueeze(-1)
    return down_new, up_new


def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target ** 2)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def to_dtype(dtype):
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_mapping.get(dtype, torch.float32)
    return dtype


def load_font():
    # Load a font
    font_path = f"{FONTS_DIR}/ShareTechMono-Regular.ttf"
    try:
        title_font = ImageFont.truetype(font_path, size=48)
    except OSError:
        logging.warning(f"PM LoRABlockSampler: Font not found at {font_path}, using default font.")
        title_font = ImageFont.load_default()
    return title_font