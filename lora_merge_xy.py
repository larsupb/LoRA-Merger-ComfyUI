import os
from abc import ABC, abstractmethod
from typing import get_args

import torch

import folder_paths
from comfy.sd import load_lora_for_models
from .lora_load import LoraPowerMergeLoader
from .lora_merge import LoraMerger, MODES
from .lora_svd_merge import LoraSVDMerger, SVD_MODES


class XYLoRAMergeCapsule(ABC):
    def __init__(self, x, y, strengths, label, storage, params):
        self.x = x
        self.y = y
        self.variables = strengths
        self.label = label
        self.storage = storage
        self.params = params
        self.another_capsule = None

    def pre_define_model(self, model, clip, vae):
        model, clip = self.patch_model(model, clip)
        return model, clip, vae

    @abstractmethod
    def patch_model(self, model, clip):
        pass

    def getLabel(self):
        # override
        return self.label

    def set_x_capsule(self, capsule):
        self.another_capsule = capsule

    def set_result(self, image, latent):
        # if self.another_capsule is not None:
        #     print(f"XY_Capsule_LoraBlockWeight: ({self.another_capsule.x, self.y}) is processed.")
        #     self.storage[(self.another_capsule.x, self.y)] = image
        # else:
        #     print(f"XY_Capsule_LoraBlockWeight: ({self.x, self.y}) is processed.")
        return None

    def get_result(self, model, clip, vae):
        return None


class XYInputPowerMergeStrengths:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_a": (folder_paths.get_filename_list("loras"),),
                "lora_b": (folder_paths.get_filename_list("loras"),),
                "mode": (["add", "concat", "ties", "dare_linear", "dare_ties", "magnitude_prune"],),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
                "min_strength": ("FLOAT", {"default": 0, "min": -0.0, "max": 1.0, "step": 0.01}),
                "max_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_strength": (["model + clip", "model", "clip"],),
                "steps": ("INT", {"default": 4, "min": 0, "max": 320, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("XY", "XY")
    RETURN_NAMES = ("X (strength a)", "Y (strength b)")

    FUNCTION = "doit"
    CATEGORY = "LoRA PowerMerge"

    def doit(self, lora_a, lora_b, mode: str, density, device, dtype,
             min_strength: float, max_strength: float, apply_strength, steps):
        strength_values = generate_floats(steps, min_strength, max_strength)
        storage = {}
        x_values, y_values = [], []
        common_params = lora_a, lora_b, mode, density, device, dtype, apply_strength

        for x, sv1 in enumerate(strength_values):
            x_values.append(
                self.XYLoRAMergeStrengthCapsule(x, 0, strength_values,
                                                f"{lora_name_pretty(lora_a)}:{sv1:.2f}", storage, common_params))

        for y, sv2 in enumerate(strength_values):
            y_values.append(
                self.XYLoRAMergeStrengthCapsule(0, y, strength_values,
                                                f"{lora_name_pretty(lora_b)}:{sv2:.2f}", storage, common_params))

        xy_type = "XY_Capsule"
        return (xy_type, x_values), (xy_type, y_values)

    class XYLoRAMergeStrengthCapsule(XYLoRAMergeCapsule):
        def patch_model(self, model, clip):
            # overrides abstract method
            merger = LoraMerger()
            lora_a, lora_b, mode, density, device, dtype, apply_strength = self.params

            loader = LoraPowerMergeLoader()
            x_model_strength = self.variables[self.another_capsule.x] if apply_strength != 'clip' else 0
            x_clip_strength = self.variables[self.another_capsule.x] if apply_strength != 'model' else 0
            y_model_strength = self.variables[self.y] if apply_strength != 'clip' else 0
            y_clip_strength = self.variables[self.y] if apply_strength != 'model' else 0

            lora_a = loader.load_lora(lora_a, x_model_strength, x_clip_strength, lbw="")[0]
            lora_b = loader.load_lora(lora_b, y_model_strength, y_clip_strength, lbw="")[0]

            lora = merger.lora_merge(lora_a, mode, density, device, dtype, lora_b=lora_b)
            model_lora, clip_lora = load_lora_for_models(model, clip, lora[0]['lora'], 1, 1)
            return model_lora, clip_lora


class XYInputPowerMergeModes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_a": ("LoRA",),
                "lora_b": ("LoRA",),
                "modes": ("STRING", {"multiline": True, "default": "add, concat, ties, dare_linear, dare_ties, magnitude_prune",
                                     "placeholder": "modes", "pysssss.autocomplete": False}),
                "min_density": ("FLOAT", {"default": 0, "min": -0.0, "max": 1.0, "step": 0.01}),
                "max_density": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "density_steps": ("INT", {"default": 4, "min": 0, "max": 320, "step": 1, "display": "number"}),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            }
        }

    RETURN_TYPES = ("XY", "XY")
    RETURN_NAMES = ("X (modes)", "Y (densities)")

    FUNCTION = "doit"
    CATEGORY = "LoRA PowerMerge"

    XY_TYPE = "XY_Capsule"

    def doit(self, lora_a, lora_b, modes: str, min_density: float, max_density: float, density_steps: int,
             device: torch.device, dtype: torch.dtype):

        mode_values = [s.strip() for s in modes.split(',') if s.strip() in get_args(MODES)]
        density_values = generate_floats(density_steps, min_density, max_density)

        x_values, y_values = [], []
        storage = {}
        common_params = lora_a, lora_b, device, dtype

        for x, sv1 in enumerate(mode_values):
            x_values.append(
                self.XYLoRAMergeModeCapsule(x, 0, (mode_values, density_values),
                                            f'mode: {sv1}', storage, common_params))
        for y, sv2 in enumerate(density_values):
            y_values.append(
                self.XYLoRAMergeModeCapsule(0, y, (mode_values, density_values),
                                            f'density: {sv2:.2f}', storage, common_params))

        return (self.XY_TYPE, x_values), (self.XY_TYPE, y_values)

    class XYLoRAMergeModeCapsule(XYLoRAMergeCapsule):
        def patch_model(self, model, clip):
            # overrides abstract method
            lora_a, lora_b, device, dtype = self.params

            mode = self.variables[0][self.another_capsule.x]
            density = self.variables[1][self.y]

            merger = LoraMerger()
            lora = merger.lora_merge(lora_a, mode, density, device, dtype, lora_b=lora_b)
            model_lora, clip_lora = load_lora_for_models(model, clip, lora[0]['lora'], 1, 1)
            return model_lora, clip_lora


class XYInputPowerMergeSVD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_a": ("LoRA",),
                "lora_b": ("LoRA",),
                "mode": (get_args(SVD_MODES),),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "min_rank": ("INT", {"default": 1, "min": 1, "max": 320, "step": 1, "display": "number"}),
                "max_rank": ("INT", {"default": 64, "min": 1, "max": 320, "step": 1, "display": "number"}),
                "rank_steps": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "display": "number"}),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X",)

    FUNCTION = "doit"
    CATEGORY = "LoRA PowerMerge"

    XY_TYPE = "XY_Capsule"

    def doit(self, lora_a, lora_b, mode: SVD_MODES, density: float,
             min_rank: int, max_rank: int, rank_steps: int,
             device: torch.device, dtype: torch.dtype):
        rank_values = generate_ints(rank_steps, min_rank, max_rank)

        x_values = []
        storage = {}
        common_params = lora_a, lora_b, mode, density, device, dtype

        for x, rv in enumerate(rank_values):
            x_values.append(
                self.XYLoRAMergeSVDCapsule(x, 0, rank_values, f'rank: {rv}', storage, common_params))

        return (self.XY_TYPE, x_values),

    class XYLoRAMergeSVDCapsule(XYLoRAMergeCapsule):
        def patch_model(self, model, clip):
            # overrides abstract method
            lora_a, lora_b, mode, density, device, dtype = self.params

            rank = self.variables[self.x]
            rank_svd = 64

            if self.x in self.storage:
                return self.storage[self.x]

            merger = LoraSVDMerger()
            lora = merger.lora_svd_merge(lora_a, mode, density, rank, rank_svd, device, lora_b=lora_b)
            model_lora, clip_lora = load_lora_for_models(model, clip, lora[0]['lora'], 1, 1)
            self.storage[self.x] = (model_lora, clip_lora)
            return model_lora, clip_lora


def generate_floats(batch_count, first_float, last_float):
    if batch_count > 1:
        interval = (last_float - first_float) / (batch_count - 1)
        return [round(first_float + i * interval, 3) for i in range(batch_count)]
    else:
        return [first_float] if batch_count == 1 else []


def generate_ints(batch_count, first_int, last_int):
    if batch_count > 1:
        interval = (last_int - first_int) / (batch_count - 1)
        values = [int(first_int + i * interval) for i in range(batch_count)]
    else:
        values = [first_int] if batch_count == 1 else []
    values = list(set(values))  # Remove duplicates
    values.sort()  # Sort in ascending order
    return values


def lora_name_pretty(lora_name):
    return os.path.splitext(lora_name)[0]
