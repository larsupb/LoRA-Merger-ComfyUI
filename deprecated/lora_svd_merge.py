from typing import Literal, get_args, List

import torch

import comfy
from .peft_utils import task_arithmetic, ties, dare_linear, dare_ties, magnitude_prune
from .architectures.sd_lora import analyse_keys, calc_up_down_alphas
from .utility import to_dtype

CLAMP_QUANTILE = 0.99
SVD_MODES = Literal["add_svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"]


class LoraSVDMerger:
    """
        Class for merging LoRA models using Singular Value Decomposition (SVD).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (get_args(SVD_MODES),),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "svd_rank": ("INT", {
                    "default": 16,
                    "min": 1,  # Minimum value
                    "max": 320,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "svd_conv_rank": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 320,
                    "step": 1,
                    "display": "number"
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_svd_merge"
    CATEGORY = "LoRA PowerMerge"

    def lora_svd_merge(self, lora1,
                       mode: SVD_MODES = "add_svd",
                       density: float = None, svd_rank: int = None, svd_conv_rank: int = None, device=None, dtype=None,
                       **kwargs):
        """
           Merge LoRA models using SVD and specified mode.

           Args:
               lora1: The first LoRA model.
               mode: The merging mode to use.
               density: The density parameter for some merging modes.
               svd_rank: The rank for SVD.
               svd_conv_rank: The convolution rank for SVD.
               device: The device to use ('cuda' or 'cpu').
               dtype: The data type for output
               **kwargs: Additional LoRA models to merge.

           Returns:
               A tuple containing the merged LoRA model.
        """
        loras = [lora1]
        for k, v in kwargs.items():
            loras.append(v)
        dtype = to_dtype(dtype)

        self.validate_input(loras, mode)

        weight = {}
        keys = analyse_keys(loras)

        pb = comfy.utils.ProgressBar(len(keys))
        for key in keys:
            # Build taskTensor weights
            strength_key = "strength_clip" if "lora_te" in key else "strength_model"
            strengths = torch.tensor([w[strength_key] for w in loras]).to(device)

            # Calculate up and down nets and their alphas
            ups_downs_alphas = calc_up_down_alphas(loras, key)

            # Build merged tensor
            weights = self.build_weights(ups_downs_alphas, strengths, mode, density, device)

            # Calculate final tensors by svd
            up, down, alpha = self.svd(weights, svd_rank, svd_conv_rank, device)

            weight[key + ".lora_up.weight"] = up.to(device='cpu', dtype=dtype)
            weight[key + ".lora_down.weight"] = down.to(device='cpu', dtype=dtype)
            weight[key + ".alpha"] = alpha.to(device='cpu', dtype=dtype)

            pb.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1}
        return (lora_out,)

    @staticmethod
    def build_weights(ups_downs_alphas, strengths, mode: SVD_MODES, density, device):
        """
            Construct the combined weight tensor from multiple LoRA up and down tensors using different
            merging modes.

            This method supports both fully connected (2D) and convolutional (4D) tensors. It scales and
            merges the up and down tensors based on the specified mode and density, performing task-specific
            arithmetic operations.

            Args:
                ups_downs_alphas (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A list of tuples,
                    where each tuple contains the up tensor, down tensor, and alpha value.
                strengths (torch.Tensor): A tensor containing the strength values for each set of up and down tensors.
                mode (Literal["svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"]):
                    The mode to use for merging the weights. Each mode applies a different method of combining the tensors.
                density (float): The density parameter used in certain modes like "ties" and "dare".
                device (torch.DeviceObjType): The device on which to perform the computations (e.g., 'cuda' or 'cpu').

            Returns:
                torch.Tensor: The combined weight tensor resulting from the specified merging process.

            Note:
                - For convolutional tensors, special handling is applied depending on the kernel size.
                - The weight tensors are scaled by their respective alpha values and normalized by their rank.
        """
        up_1, down_1, alpha_1 = ups_downs_alphas[0]
        conv2d = len(down_1.size()) == 4
        kernel_size = None if not conv2d else down_1.size()[2:4]

        # lora = up @ down * alpha / rank
        weights = []
        for up, down, alpha in ups_downs_alphas:
            up, down, alpha = up.to(device), down.to(device), alpha.to(device)
            rank = up.shape[1]
            if conv2d:
                if kernel_size == (1, 1):
                    weight = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                        3) * alpha / rank
                else:
                    weight = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3) * alpha / rank
            else:  # linear
                weight = up.view(-1, rank) @ down.view(rank, -1) * alpha / rank
            weights.append(weight)

        if mode == "add_svd":
            weight = task_arithmetic(weights, strengths)
        elif mode == "ties_svd":
            weight = ties(weights, strengths, density)
        elif mode == "dare_linear_svd":
            weight = dare_linear(weights, strengths, density)
        elif mode == "dare_ties_svd":
            weight = dare_ties(weights, strengths, density)
        else:  # mode == "magnitude_prune_svd":
            weight = magnitude_prune(weights, strengths, density)

        return weight

    @staticmethod
    def svd(weights: torch.Tensor, svd_rank: int, svd_conv_rank: int, device: str):
        """
            Perform Singular Value Decomposition (SVD) on the given weights tensor and return the
            decomposed matrices with the specified ranks.

            This method supports both 2D (fully connected) and 4D (convolutional) weight tensors. For
            convolutional tensors, it handles both 1x1 and other kernel sizes. The ranks for decomposition
            are adjusted based on the input tensor's dimensions and the specified rank constraints.

            Args:
                weights (torch.Tensor): The input weight tensor to decompose. Should be either a 2D or 4D tensor.
                svd_rank (int): The rank for SVD decomposition for fully connected layers.
                svd_conv_rank (int): The rank for SVD decomposition for convolutional layers with kernel sizes other than 1x1.
                device (torch.DeviceObjType): The device on which to perform the computations (e.g., 'cuda' or 'cpu').

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                    - up_weight: The U matrix after SVD decomposition, representing the left singular vectors.
                    - down_weight: The Vh matrix after SVD decomposition, representing the right singular vectors.
                    - module_new_rank: A tensor containing the new rank used for the decomposition.

            Note:
                SVD only supports float32 data type, so the input weights tensor is converted to float32 if necessary.
        """
        weights = weights.to(dtype=torch.float32, device=device)  # SVD only supports float32

        conv2d = len(weights.size()) == 4
        kernel_size = None if not conv2d else weights.size()[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)
        out_dim, in_dim = weights.size()[0:2]

        if conv2d:
            if conv2d_3x3:
                weights = weights.flatten(start_dim=1)
            else:
                weights = weights.squeeze()

        module_new_rank = svd_conv_rank if conv2d_3x3 else svd_rank
        module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

        U, S, Vh = torch.linalg.svd(weights)

        U = U[:, :module_new_rank]
        S = S[:module_new_rank]
        U = U @ torch.diag(S)

        Vh = Vh[:module_new_rank, :]

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if conv2d:
            U = U.reshape(out_dim, module_new_rank, 1, 1)
            Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

        up_weight = U
        down_weight = Vh

        return up_weight, down_weight, torch.tensor(module_new_rank)

    @staticmethod
    def validate_input(loras, mode):
        if mode not in get_args(SVD_MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")
