import torch


def find_network_dim(lora_sd: dict):
    network_dim = None
    for key, value in lora_sd.items():
        if network_dim is None and ('lora_down' in key or 'lora_A' in key) and len(value.size()) == 2:
            network_dim = value.size()[0]
    return network_dim


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
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
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