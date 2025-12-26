from typing import Dict, Literal

import torch
from mergekit.common import ModelReference

MERGEKIT_GTA_MODES = Literal[
    "della", "breadcrumbs", "dare", "ties", "task_arithmetic", "linear"]


def load_on_device(tensors: Dict[ModelReference, torch.Tensor],
                   tensor_weights: Dict[ModelReference, torch.Tensor], device, dtype):
    """
    Load tensors onto device with specified dtype.
    Ensures tensors are contiguous to avoid mergekit sparsify errors.
    """
    for k, v in tensors.items():
        # Ensure contiguous before moving to device to avoid .view() errors in mergekit
        tensors[k] = v.contiguous().to(device=device, dtype=dtype)
    for k, v in tensor_weights.items():
        tensor_weights[k] = v.contiguous().to(device=device, dtype=dtype)
