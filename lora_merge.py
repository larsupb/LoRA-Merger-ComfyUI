import math
from typing import Literal, get_args

import torch
import comfy

# Assuming these are in a local utility file
from .peft_utils import task_arithmetic, ties, dare_linear, dare_ties, magnitude_prune, concat
from .utility import to_dtype

CLAMP_QUANTILE = 0.99
MODES = Literal["add", "concat", "ties", "dare_linear", "dare_ties", "magnitude_prune"]

# --- UPDATED: Added Randomized SVD Modes ---
SVD_MODES = Literal["add_svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"]
RANDOM_SVD_MODES = Literal[
    "add_random_svd", "ties_random_svd", "dare_linear_random_svd", "dare_ties_random_svd", "magnitude_prune_random_svd"]
ALL_SVD_MODES = get_args(SVD_MODES) + get_args(RANDOM_SVD_MODES)


# --- END UPDATE ---


def find_network_dim(lora_dict):
    """
    Finds the rank/dimension of a LoRA network.
    Handles both up/down and A/B naming conventions.
    """
    for key, weight in lora_dict.items():
        if ".lora_down.weight" in key or ".lora_A.weight" in key:
            return weight.shape[0]
    print("Warning: Could not determine LoRA rank.")
    return 0


def get_naming_convention(lora_dict):
    """
    Determines the naming convention of a LoRA's weights based on its keys.
    """
    for key in lora_dict.keys():
        if ".lora_A.weight" in key:
            return ".lora_B.weight", ".lora_A.weight"
    return ".lora_up.weight", ".lora_down.weight"


class LoraMerger:
    """
       Class for merging LoRA models of the same rank.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (get_args(MODES),),
                "density": ("FLOAT", {
                    "default": 1.0, "min": 0, "max": 1, "step": 0.01,
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_merge"
    CATEGORY = "LoRA PowerMerge"

    @torch.no_grad()
    def lora_merge(self, lora1,
                   mode: MODES = None,
                   density=None, device=None, dtype=None, **kwargs):
        loras = [lora1] + [v for k, v in kwargs.items()]
        self.validate_input(loras, mode)
        up_suffix, down_suffix = get_naming_convention(lora1['lora'])
        dtype = to_dtype(dtype)
        keys = analyse_keys(loras)
        weight = {}
        pbar = comfy.utils.ProgressBar(len(keys))

        for key in keys:
            scale_key = "strength_clip" if "lora_te" in key else "strength_model"
            weights = torch.tensor([w[scale_key] for w in loras]).to(device, dtype=dtype)
            ups_downs_alphas = calc_up_down_alphas(loras, key)
            ups_downs_alphas, alpha_1 = scale_alphas(ups_downs_alphas)
            ups_downs_alphas = curate_tensors(ups_downs_alphas)
            up_tensors = [up.to(device, dtype=dtype) for up, down, alpha in ups_downs_alphas]
            down_tensors = [down.to(device, dtype=dtype) for up, down, alpha in ups_downs_alphas]

            if mode == "add":
                up, down = (task_arithmetic(up_tensors, weights), task_arithmetic(down_tensors, weights))
            elif mode == "concat":
                up, down = (concat(up_tensors, weights, dim=1), concat(down_tensors, weights, dim=0))
            elif mode == "ties":
                up, down = (ties(up_tensors, weights, density), ties(down_tensors, weights, density))
            elif mode == "dare_linear":
                up, down = (dare_linear(up_tensors, weights, density), dare_linear(down_tensors, weights, density))
            elif mode == "dare_ties":
                up, down = (dare_ties(up_tensors, weights, density), dare_ties(down_tensors, weights, density))
            else:  # magnitude_prune
                up, down = (magnitude_prune(up_tensors, weights, density),
                            magnitude_prune(down_tensors, weights, density))

            weight[key + up_suffix] = up.to('cpu', dtype=torch.float32)
            weight[key + down_suffix] = down.to('cpu', dtype=torch.float32)
            weight[key + ".alpha"] = alpha_1.to('cpu', dtype=torch.float32)
            pbar.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1,
                    "name": "merged_of_" + "_".join([l['name'] for l in loras])}
        return (lora_out,)

    def validate_input(self, loras, mode):
        dims = [find_network_dim(lora['lora']) for lora in loras]
        if min(dims) != max(dims):
            raise Exception("LoRAs with different ranks not allowed in LoraMerger. Use SVD merge.")
        if mode not in get_args(MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")


class LoraSVDMerger:
    """
        Class for merging LoRA models using Singular Value Decomposition (SVD).
        Now supports standard and randomized SVD.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (ALL_SVD_MODES,),  # --- UPDATED: Use combined list of modes ---
                "density": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
                "svd_rank": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 1, "display": "number"}),
                "svd_conv_rank": ("INT", {"default": 1, "min": 0, "max": 1024, "step": 1, "display": "number"}),
                # --- NEW: Input for randomized SVD iterations ---
                "random_n_iter": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1, "display": "number"}),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_svd_merge"
    CATEGORY = "LoRA PowerMerge"

    # --- UPDATED: Signature includes new parameter `random_n_iter` ---
    def lora_svd_merge(self, lora1,
                       mode: str = "add_svd",
                       density: float = None, svd_rank: int = None, svd_conv_rank: int = None,
                       random_n_iter: int = 4,
                       device=None, dtype=None,
                       **kwargs):
        loras = [lora1] + [v for k, v in kwargs.items()]
        dtype = to_dtype(dtype)
        self.validate_input(loras, mode)

        up_suffix, down_suffix = get_naming_convention(lora1['lora'])
        weight = {}
        keys = analyse_keys(loras)
        pb = comfy.utils.ProgressBar(len(keys))

        # --- UPDATED: Logic to determine SVD type and base mode ---
        is_random_svd = "random" in mode
        base_mode = mode.replace('_random', '')

        for key in keys:
            strength_key = "strength_clip" if "lora_te" in key else "strength_model"
            strengths = torch.tensor([w[strength_key] for w in loras]).to(device)
            ups_downs_alphas = calc_up_down_alphas(loras, key, fill_with_empty_tensor=True)

            weights = self.build_weights(ups_downs_alphas, strengths, base_mode, density, device)

            # --- UPDATED: Pass SVD type and params to the svd method ---
            up, down, alpha = self.svd(weights, svd_rank, svd_conv_rank, device,
                                       randomized=is_random_svd, n_iter=random_n_iter)

            weight[key + up_suffix] = up.to(device='cpu', dtype=torch.float32)
            weight[key + down_suffix] = down.to(device='cpu', dtype=torch.float32)
            weight[key + ".alpha"] = alpha.to(device='cpu', dtype=torch.float32)
            pb.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1}
        return (lora_out,)

    def validate_input(self, loras, mode):
        # --- UPDATED: Check against all SVD modes ---
        if mode not in ALL_SVD_MODES:
            raise Exception(f"Invalid / unsupported mode {mode}")

    def build_weights(self, ups_downs_alphas, strengths,
                      mode: str, density, device):
        up_1, down_1, alpha_1 = ups_downs_alphas[0]
        conv2d = len(down_1.size()) == 4
        kernel_size = None if not conv2d else down_1.size()[2:4]
        weights = []

        for up, down, alpha in ups_downs_alphas:
            up, down, alpha = up.to(device), down.to(device), alpha.to(device)
            rank = down.shape[0]
            if rank == 0: continue  # Skip empty tensors

            if conv2d:
                if kernel_size == (1, 1):
                    w = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                else:
                    w = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                w = up @ down
            weights.append(w * (alpha / rank))

        if mode == "add_svd":
            weight = task_arithmetic(weights, strengths)
        elif mode == "ties_svd":
            weight = ties(weights, strengths, density)
        elif mode == "dare_linear_svd":
            weight = dare_linear(weights, strengths, density)
        elif mode == "dare_ties_svd":
            weight = dare_ties(weights, strengths, density)
        else:  # magnitude_prune_svd
            weight = magnitude_prune(weights, strengths, density)

        return weight

    # --- MAJOR UPDATE: SVD method now handles both standard and randomized SVD ---
    def svd(self, weights: torch.Tensor, svd_rank: int, svd_conv_rank: int, device: str,
            randomized: bool = False, n_iter: int = 4):
        """
        Perform Singular Value Decomposition (SVD) on the given weights tensor.
        Can use standard SVD or randomized SVD for faster approximation.
        """
        weights = weights.to(dtype=torch.float32, device=device)
        conv2d = len(weights.size()) == 4
        kernel_size = None if not conv2d else weights.size()[2:4]
        out_dim, in_dim = weights.size()[0:2]

        # Determine target rank for this module
        module_new_rank = svd_conv_rank if conv2d and kernel_size != (1, 1) else svd_rank
        module_new_rank = min(module_new_rank, in_dim, out_dim)

        # Handle rank=0 case gracefully
        if module_new_rank == 0:
            final_U_shape = (out_dim, 0, 1, 1) if conv2d else (out_dim, 0)
            final_Vh_shape = (0, in_dim, *kernel_size) if conv2d else (0, in_dim)
            return (torch.zeros(final_U_shape),
                    torch.zeros(final_Vh_shape),
                    torch.tensor(0.0))

        # Flatten conv layers for SVD
        if conv2d:
            if kernel_size != (1, 1):
                weights = weights.flatten(start_dim=1)
            else:
                weights = weights.squeeze()

        if randomized:
            # Use randomized SVD (torch.pca_lowrank) for speed
            # q must be <= min(A.shape), which module_new_rank is.
            U, S, V = torch.pca_lowrank(weights, q=module_new_rank, center=False, niter=n_iter)
            Vh = V.T  # pca_lowrank returns V, so we transpose to get Vh
        else:
            # Use standard, full SVD
            U, S, Vh = torch.linalg.svd(weights, full_matrices=False)

        # Truncate and combine S into U
        U = U[:, :module_new_rank]
        S = S[:module_new_rank]
        Vh = Vh[:module_new_rank, :]
        U = U @ torch.diag(S)

        # Clamp values to prevent outliers
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        # Reshape back to conv kernels if necessary
        if conv2d:
            U = U.reshape(out_dim, module_new_rank, 1, 1)
            Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

        return U, Vh, torch.tensor(float(module_new_rank))


@torch.no_grad()
def calc_up_down_alphas(loras, key, fill_with_empty_tensor=False):
    alpha_key = key + ".alpha"
    ref_lora_data = None
    for lora in loras:
        lora_dict = lora['lora']
        if f"{key}.lora_down.weight" in lora_dict or f"{key}.lora_A.weight" in lora_dict:
            ref_lora_data = lora_dict
            break
    if not ref_lora_data: return []

    down_key_ref = f"{key}.lora_A.weight" if f"{key}.lora_A.weight" in ref_lora_data else f"{key}.lora_down.weight"
    up_key_ref = f"{key}.lora_B.weight" if f"{key}.lora_A.weight" in ref_lora_data else f"{key}.lora_up.weight"
    down_shape, up_shape = ref_lora_data[down_key_ref].shape, ref_lora_data[up_key_ref].shape

    rank_ref = down_shape[0]
    alpha_1_val = ref_lora_data.get(alpha_key, rank_ref)
    if not isinstance(alpha_1_val, torch.Tensor):
        alpha_1_val = torch.tensor(float(alpha_1_val))

    out = []
    for lora in loras:
        lora_dict = lora['lora']
        down_key = f"{key}.lora_A.weight" if f"{key}.lora_A.weight" in lora_dict else f"{key}.lora_down.weight"
        up_key = f"{key}.lora_B.weight" if f"{key}.lora_A.weight" in lora_dict else f"{key}.lora_up.weight"

        if up_key in lora_dict:
            up, down = lora_dict[up_key], lora_dict[down_key]
            rank = down.shape[0]
            alpha = lora_dict.get(alpha_key, rank)
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(float(alpha), device=up.device)
            out.append((up, down, alpha))
        elif fill_with_empty_tensor:
            up_placeholder, down_placeholder = torch.zeros(up_shape), torch.zeros(down_shape)
            out.append((up_placeholder, down_placeholder, alpha_1_val.clone()))
    return out


def scale_alphas(ups_downs_alphas):
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = []
    for up, down, alpha in ups_downs_alphas:
        if alpha_1 != 0:
            scale = math.sqrt(alpha / alpha_1)
            up = up * scale
            down = down * scale
        out.append((up, down, alpha_1))
    return out, alpha_1


def analyse_keys(loras):
    module_keys = set()
    for i, lora in enumerate(loras):
        key_count = 0
        lora_name = lora.get('name', f'#{i + 1}')
        for key in lora["lora"].keys():
            if ".lora_down.weight" in key or ".lora_A.weight" in key:
                base_key = key.rsplit('.', 2)[0]
                module_keys.add(base_key)
                key_count += 1
        print(f"LoRA '{lora_name}' has {key_count} modules.")
    print(f"Found {len(module_keys)} unique modules to merge.")
    return module_keys


def curate_tensors(ups_downs_alphas):
    if not ups_downs_alphas: return []
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = [ups_downs_alphas[0]]
    for up, down, alpha in ups_downs_alphas[1:]:
        up = adjust_tensor_to_match(up_1, up)
        down = adjust_tensor_to_match(down_1, down)
        out.append((up, down, alpha))
    return out


def adjust_tensor_to_match(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    shape1, shape2 = tensor1.shape, tensor2.shape
    if shape1 == shape2: return tensor2
    adjusted_tensor = torch.zeros(shape1, dtype=tensor2.dtype, device=tensor2.device)
    slices = tuple(slice(0, min(d1, d2)) for d1, d2 in zip(shape1, shape2))
    adjusted_tensor[slices] = tensor2[slices]
    return adjusted_tensor