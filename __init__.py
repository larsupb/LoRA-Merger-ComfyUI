from .checkpoint_merge import CheckpointMergerMergekit
from .lora_apply import LoraApply
from .lora_attention_logger import LoRAAttentionLogger, LoRAAttentionPlot
from .lora_block_sampler import LoRABlockSampler
from .lora_stack_sampler import LoRAStackSampler
from .lora_load import LoraPowerMergeLoader
from .lora_merge_xy import XYInputPowerMergeStrengths, XYInputPowerMergeModes, XYInputPowerMergeSVD
from .lora_mergekit_merge import LoraMergerMergekit, LoraStack, LoraDecompose, LoraStackFromDir, LoRASelect
from .lora_resize import LoraResizer
from .lora_save import LoraSave
from .nodes_experimental_methods import LoRAPrune, LoRAAnalyzer, LoRAModifier
from .nodes_merge_methods import TaskArithmeticMergeMethod, NearSwapMergeMethod, SCEMergeMethod, BreadcrumbsMergeMethod, \
    TIESMergeMethod, DAREMergeMethod, DELLAMergeMethod, SLERPMergeMethod, LinearMergeMethod, NuSlerpMergeMethod, \
    ArceeFusionMergeMethod, KArcherMergeMethod

version_code = [0, 11]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: ComfyUI LoRA-PowerMerge ({version_str})")

NODE_CLASS_MAPPINGS = {
    "PM LoRA Merger (Mergekit)": LoraMergerMergekit,

    "PM LoRA Stacker": LoraStack,
    "PM LoRA Stacker (from Directory)": LoraStackFromDir,
    "PM LoRA Select": LoRASelect,
    "PM LoRA Stack Decompose": LoraDecompose,

    "PM Checkpoint Merger (Mergekit)": CheckpointMergerMergekit,

    "PM LoRA Block Sampler": LoRABlockSampler,
    "PM LoRA Stack Sampler": LoRAStackSampler,

    "PM Slerp (Mergekit)": SLERPMergeMethod,
    "PM NuSlerp (Mergekit)": NuSlerpMergeMethod,
    "PM NearSwap (Mergekit)": NearSwapMergeMethod,
    "PM Arcee Fusion (Mergekit)": ArceeFusionMergeMethod,

    "PM Linear (Mergekit)": LinearMergeMethod,
    "PM SCE (Mergekit)": SCEMergeMethod,
    "PM KArcher (Mergekit)": KArcherMergeMethod,

    "PM Task Arithmetic (Mergekit)": TaskArithmeticMergeMethod,
    "PM Ties (Mergekit)": TIESMergeMethod,
    "PM Breadcrumbs (Mergekit)": BreadcrumbsMergeMethod,
    "PM Dare (Mergekit)": DAREMergeMethod,
    "PM Della (Mergekit)": DELLAMergeMethod,

    "PM LoRA Analyzer": LoRAAnalyzer,
    "PM LoRA AttentionLogger": LoRAAttentionLogger,
    "PM LoRA Attention Plot": LoRAAttentionPlot,

    "PM LoRA Modifier": LoRAModifier,
    "PM LoRA Prune": LoRAPrune,

    "PM LoRA Resizer": LoraResizer,
    "PM LoRA Apply": LoraApply,
    "PM LoRA Loader": LoraPowerMergeLoader,
    "PM LoRA Save": LoraSave,

    "XY: PM LoRA Strengths": XYInputPowerMergeStrengths,
    "XY: PM LoRA Modes": XYInputPowerMergeModes,
    "XY: PM LoRA SVD Rank": XYInputPowerMergeSVD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PM LoRA Stacker": "PM LoRA Stacker",
    "PM LoRA Stacker (from Directory)": "PM LoRA Stacker (from Directory)",
    "PM LoRA Select": "PM LoRA Select",
    "PM LoRA Stack Decompose": "PM LoRA Stack Decompose",

    "PM LoRA Merger (Mergekit)": "PM LoRA Merger (Mergekit)",
    "PM Checkpoint Merger (Mergekit)": "PM Checkpoint Merger (Mergekit)",

    "PM LoRA Block Sampler": "PM LoRA Block Sampler",
    "PM LoRA Stack Sampler": "PM LoRA Stack Sampler",

    "PM Slerp (Mergekit)": "PM Slerp (Mergekit)",
    "PM NuSlerp (Mergekit)": "PM NuSlerp (Mergekit)",
    "PM NearSwap (Mergekit)": "PM NearSwap (Mergekit)",
    "PM Arcee Fusion (Mergekit)": "PM Arcee Fusion (Mergekit)",

    "PM Linear (Mergekit)": "PM Linear (Mergekit)",
    "PM SCE (Mergekit)": "PM SCE (Mergekit)",
    "PM KArcher (Mergekit)": "PM KArcher (Mergekit)",

    "PM Task Arithmetic (Mergekit)": "PM Task Arithmetic (Mergekit)",
    "PM TIES (Mergekit)": "PM TIES (Mergekit)",
    "PM DARE (Mergekit)": "PM DARE (Mergekit)",
    "PM Breadcrumbs (Mergekit)": "PM Breadcrumbs (Mergekit)",
    "PM Della (Mergekit)": "PM Della (Mergekit)",

    "PM LoRA Analyzer": "PM LoRA Analyzer",
    "PM LoRA AttentionLogger": "PM LoRA Attention Logger",
    "PM LoRA Attention Plot": "PM LoRA Attention Plot",

    "PM LoRA Modifier": "PM LoRA Modifier",
    "PM LoRA Prune": "PM LoRA Prune",

    "PM LoRA Resizer": "PM Resize LoRA",
    "PM LoRA Apply": "PM Apply LoRA",
    "PM LoRA Loader": "PM Load LoRA",
    "PM LoRA Save": "PM Save LoRA",

    "XY: PM LoRA Strengths": "XY: LoRA Power-Merge Strengths",
    "XY: PM LoRA Modes": "XY: LoRA Power-Merge Modes",
    "XY: PM LoRA SVD Rank": "XY: LoRA Power-Merge SVD Rank",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
