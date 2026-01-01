from .merge import BaseMergeMethodNode


class LinearMergeMethod(BaseMergeMethodNode):
    """
    Linear merge method node.

    Concept: Computes a simple weighted average of the parameters from the input models.
    This is one of the most basic and widely used merging techniques.

    Use Cases:
    - Averaging multiple checkpoints of the same fine-tuning run ("model soups")
    - Combining models with very similar architectures and training data
    - Simple ensemble-like behavior in a single model

    Inputs: Takes 2 or more models. No base_model is typically used.

    Key Parameters:
    - weight (per-model): The contribution of each model to the average
    - normalize (global): If true (default), weights are normalized to sum to 1
    """

    METHOD_NAME = "linear"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized. Default behavior.",
                }),
            },
        }

    def get_settings(self, normalize: bool = True):
        return {"normalize": normalize}


class SCEMergeMethod(BaseMergeMethodNode):
    """
    SCE (Select, Calculate, Erase) merge method node.

    Concept: The SCE method performs adaptive matrix-level merging.
    It first computes task vectors (differences from the base_model). Then, it follows
    a three-step process for each parameter matrix (tensor):

    1. Select (Variance-Based Masking): Optionally, parameter positions that show low
       variance across the different models' task vectors are identified and zeroed out.
       This is controlled by the select_topk parameter.

    2. Calculate (Weighting): Matrix-level merging coefficients (weights) are calculated
       for each model's task vector.

    3. Erase (Sign Consensus): The sign-consensus algorithm from TIES is applied to the
       task vectors.

    Use Cases:
    - Dynamically weighting the contribution of different models at the matrix level
    - Useful when some models contribute more significantly to certain parameter matrices
    - Merging models by focusing on high-variance, consistently signed changes

    Inputs: Requires 2 or more models, plus one base_model.

    Key Parameters:
    - select_topk (global): The fraction of parameter positions to retain based on their
      variance values across the different input models' task vectors.
    """

    METHOD_NAME = "sce"
    CATEGORY = "LoRA PowerMerge/Task Arithmetic"
    DESCRIPTION = """SCE (Select, Calculate, Erase) adaptive merge method.

Three-stage merge process:
1. SELECT: Variance-based masking - zeros out low-variance parameters (select_topk controls retention)
2. CALCULATE: Matrix-level weight computation for each LoRA
3. ERASE: Sign consensus algorithm (from TIES) to eliminate conflicting changes

Best for: Merging models where different LoRAs contribute to different parameter matrices. Dynamically weights contributions based on variance and sign consensus.

Parameters:
- select_topk: Fraction of highest-variance elements to retain (0.1 = keep top 10%)

Requires: 2+ LoRAs plus base model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_topk": ("FLOAT", {
                    "default": .1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Fraction of elements with the highest variance in the delta parameters to retain.",
                }),
            },
        }

    def get_settings(self, select_topk: float = 0.1):
        return {"select_topk": select_topk}


class TaskArithmeticMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rescale_norm": (["default", "l1", "l2", "linf", "none"],
                                 {"default": "default", "tooltip": "Rescaling strategy:\n"
                                  "• default: Auto-select (L1 for methods needing it, none otherwise)\n"
                                  "• l1: L1 norm preservation (precise, preserves magnitude sum)\n"
                                  "• l2: L2 norm preservation (precise, preserves Euclidean norm)\n"
                                  "• linf: L-infinity norm (preserves max absolute value, prevents amplification)\n"
                                  "• none: No rescaling (may reduce merge strength)"}),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized. Default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Task Arithmetic"
    DESCRIPTION = """
Concept: Computes "task vectors" for each model by subtracting a base_model. 
These task vectors are then combined as a weighted average and added back to the base_model.

Use Cases:    
Combining skills from multiple models fine-tuned from a common ancestor
Transferring specific capabilities (e.g., coding ability, instruction following) from one model to another
Steering style or behavior of a model by adding small task vectors from other models
Inputs: Requires a base_model and one or more other models.

Key Parameters:    
- weight (per-model): Weight for each model's task vector in the merge
- lambda (global): Scaling factor applied to the summed task vectors before adding back to the base. Default 1.0
"""

    def get_method(self, rescale_norm: str = "default", normalize: bool = 0.5, ):
        method_def = {
            "name": "task_arithmetic",
            "settings": {
                "rescale_norm": rescale_norm,
                "normalize": normalize,
            }
        }
        return (method_def,)


class TIESMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rescale_norm": (["default", "l1", "l2", "linf", "none"],
                                 {"default": "default", "tooltip": "Rescaling strategy:\n"
                                  "• default: Auto-select (L1 for methods needing it, none otherwise)\n"
                                  "• l1: L1 norm preservation (precise, preserves magnitude sum)\n"
                                  "• l2: L2 norm preservation (precise, preserves Euclidean norm)\n"
                                  "• linf: L-infinity norm (preserves max absolute value, prevents amplification)\n"
                                  "• none: No rescaling (may reduce merge strength)"}),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized. Default behavior.",
                }),
                "density": ("FLOAT", {
                    "default": .9,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Fraction of weights in differences from the base model to retain.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Task Arithmetic"
    DESCRIPTION = """
    Concept: Builds on Task Arithmetic by sparsifying task vectors and applying a sign consensus algorithm. 
    This helps to resolve interference when merging multiple models and retain more of their individual strengths.

    Use Cases:    
    Merging a larger number of models effectively
    Reducing parameter interference and negative synergy between merged models
    Inputs: Requires 2 or more models.
    
    Key Parameters:
    - weight (per-model): Weight for each model's task vector
    - density (per-model): Fraction of weights to retain in each sparsified task vector
    - lambda (global): As in Task Arithmetic
    """

    def get_method(self, rescale_norm:str = "default", normalize: bool = True, density: float = 1):
        method_def = {
            "name": "ties",
            "settings": {
                "rescale_norm": rescale_norm,
                "normalize": normalize,
                "density": density,
            }
        }
        return (method_def,)


class BreadcrumbsMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ties": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use sign consensus algorithm of ties algorithm.",
                }),
                "rescale_norm": (["default", "l1", "l2", "linf", "none"],
                                 {"default": "default", "tooltip": "Rescaling strategy:\n"
                                  "• default: Auto-select (L1 for methods needing it, none otherwise)\n"
                                  "• l1: L1 norm preservation (precise, preserves magnitude sum)\n"
                                  "• l2: L2 norm preservation (precise, preserves Euclidean norm)\n"
                                  "• linf: L-infinity norm (preserves max absolute value, prevents amplification)\n"
                                  "• none: No rescaling (may reduce merge strength)"}),
                "density": ("FLOAT", {
                    "default": .9,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Fraction of weights in differences from the base model to retain.",
                }),
                "gamma": ("FLOAT", {
                    "default": .1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Fraction of largest magnitude differences to remove",
                }),
                "normalize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized. "
                               "Default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Task Arithmetic"
    DESCRIPTION = """
    Model Breadcrumbs (breadcrumbs, breadcrumbs_ties)
    Concept: An extension of task arithmetic designed to sparsify task vectors by pruning parameters with both 
    the smallest and the largest absolute magnitudes (often considered outliers). This method operates in two main 
    steps on the task vector (the difference between a fine-tuned model and the base_model):

    First, a gamma fraction of the parameters with the largest absolute magnitudes are identified for removal.
    Then, parameters with the smallest absolute magnitudes are identified for removal. The quantity of these smallest 
    parameters to remove is determined such that the final density of parameters retained in the task vector is 
    achieved, after accounting for the largest ones removed.
    The intention is to isolate and merge the "meaty," mid-range magnitude changes from the task vector, potentially 
    filtering out noise (smallest changes) and overly dominant or conflicting large changes (largest changes).

    Variants:

    breadcrumbs: Model Breadcrumbs pruning without TIES sign consensus
    breadcrumbs_ties: Model Breadcrumbs pruning with TIES sign consensus
    Use Cases:

    Merging models where extreme parameter changes might be detrimental or noisy
    Refining task vectors by focusing on mid-range modifications, removing both the least significant and most 
    extreme changes

    Inputs: Requires 2 or more models.

    Key Parameters:

    weight (per-model): 
        Weight for each model's task vector.
        gamma (per-model): The fraction of parameters with the largest absolute magnitudes in the task vector 
        to be pruned (removed). For example, a gamma of 0.01 targets the removal of the top 1% of parameters 
        with the highest absolute values. This parameter corresponds to β (beta) as described in the reference paper.

    density (per-model): 
        The final target fraction of parameters to retain in the task vector after both 
        pruning steps (removal of largest gamma fraction and a corresponding fraction of smallest magnitude parameters).
        The fraction of parameters with the smallest absolute magnitudes that will be pruned is calculated 
        based on density and gamma. Specifically, it is max(0, 1.0 - density - gamma).

        Example: If density: 0.9 and gamma: 0.01:
        The top 0.01 (1%) largest magnitude parameters are removed.
        The bottom 1.0 - 0.9 - 0.01 = 0.09 (9%) smallest magnitude parameters are also removed.
        This results in 0.9 (90%) of the parameters being retained.

        Edge Case: If gamma is set high enough such that gamma >= 1.0 - density (meaning 1.0 - density - gamma <= 0), 
        then the number of largest magnitude parameters actually pruned will be adjusted to 1.0 - density, 
        and no smallest magnitude parameters will be pruned (i.e., the fraction of smallest parameters pruned becomes 0). 
        This ensures the density target is always respected and represents the fraction of parameters kept.

    lambda (global): 
        As in Task Arithmetic.
    """

    def get_method(self, ties: bool = False, rescale_norm: str = "default", density: float = .9, gamma: float = .1,
                   normalize: bool = False):
        method_def = {
            "name": "breadcrumbs",
            "settings": {
                "rescale_norm": rescale_norm,
                "normalize": normalize,
                "density": density,
                "gamma": gamma,
                "sign_consensus_algorithm": ties,
            }
        }
        return (method_def,)


class DAREMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ties": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use sign consensus algorithm of ties algorithm.",
                }),
                "rescale_norm": (["default", "l1", "l2", "linf", "none"],
                                 {"default": "default", "tooltip": "Rescaling strategy:\n"
                                  "• default: Auto-select (L1 for methods needing it, none otherwise)\n"
                                  "• l1: L1 norm preservation (precise, preserves magnitude sum)\n"
                                  "• l2: L2 norm preservation (precise, preserves Euclidean norm)\n"
                                  "• linf: L-infinity norm (preserves max absolute value, prevents amplification)\n"
                                  "• none: No rescaling (may reduce merge strength)"}),
                "density": ("FLOAT", {
                    "default": .85,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Fraction of weights to retain after random pruning.",
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized. Default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Task Arithmetic"
    DESCRIPTION = """
    Concept: Similar to TIES, DARE sparsifies task vectors to reduce interference. However, DARE uses 
    random pruning with a novel rescaling technique to better match the performance of the original models.

    Variants:
    - dare_linear: DARE pruning without the TIES sign consensus
    - dare_ties: DARE pruning with the TIES sign consensus
    
    Use Cases:    
    - Robustly combining multiple fine-tuned models, often yielding better performance than TIES in some scenarios
    - Inputs: Requires 2 or more models.
    
    Key Parameters:    
    - weight (per-model): Weight for each model's task vector
    - density (per-model): Fraction of weights to retain after random pruning
    - lambda (global): As in Task Arithmetic
    - rescale (global, for dare_linear): If true (default), applies DARE's rescaling
    """

    def get_method(self, ties: bool = False, rescale_norm: str = "default", density: float = .8,
                   normalize: bool = True):
        method_def = {
            "name": "dare",
            "settings": {
                "rescale_norm": rescale_norm,
                "normalize": normalize,
                "density": density,
                "sign_consensus_algorithm": ties,
            }
        }
        return (method_def,)


class DELLAMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ties": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use sign consensus algorithm of ties algorithm.",
                }),
                "rescale_norm": (["default", "l1", "l2", "linf", "none"],
                                 {"default": "default", "tooltip": "Rescaling strategy:\n"
                                  "• default: Auto-select (L1 for methods needing it, none otherwise)\n"
                                  "• l1: L1 norm preservation (precise, preserves magnitude sum)\n"
                                  "• l2: L2 norm preservation (precise, preserves Euclidean norm)\n"
                                  "• linf: L-infinity norm (preserves max absolute value, prevents amplification)\n"
                                  "• none: No rescaling (may reduce merge strength)"}),
                "density": ("FLOAT", {
                    "default": .85,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Fraction of weights in differences from the base model to retain.",
                }),
                "epsilon": ("FLOAT", {
                    "default": 0.1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Maximum change in drop probability based on magnitude. "
                               "Drop probabilities assigned will range from density - epsilon to density + epsilon.",
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the weights of all models contributing to a tensor will be normalized. Default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Task Arithmetic"
    DESCRIPTION = """
    Concept: Extends DARE by using adaptive pruning based on parameter magnitudes within each row of the delta 
    parameters (task vectors). 
    It calculates keep probabilities for each parameter: parameters with larger magnitudes within a row 
    are assigned higher probabilities of being kept, while parameters with smaller magnitudes are assigned lower 
    probabilities. 
    These keep probabilities are scaled to range from 
    density - epsilon (for the smallest magnitude element in a row) 
    to 
    density + epsilon (for the largest magnitude element in a row). 
    This method aims to retain important changes while reducing interference, followed by DARE-like rescaling.

    Variants:

    della: DELLA pruning with TIES sign consensus
    della_linear: DELLA pruning without TIES sign consensus
    Use Cases:

    Fine-grained control over pruning by prioritizing parameters with larger magnitude changes
    Combining models where preserving the most significant changes is crucial
    Inputs: Requires 2 or more models, plus one base_model.

    Key Parameters:

    weight (per-model): Weight for each model's task vector
    density (per-model): Target fraction of weights to retain in differences from the base model
    epsilon (per-model): Defines the half-width of the range for keep probabilities. 
    Keep probabilities for parameters in a row will range from density - epsilon to density + epsilon, 
    mapped from the smallest to largest magnitude parameters in that row, respectively. 
    epsilon must be chosen such that density - epsilon > 0 and density + epsilon < 1.
    lambda (global): As in Task Arithmetic
    """

    def get_method(self, ties: bool = False, rescale_norm: str = "default", density: float = 1, epsilon: float = .01,
                   normalize: bool = True):
        method_def = {
            "name": "della",
            "settings": {
                "rescale_norm": rescale_norm,
                "normalize": normalize,
                "density": density,
                "epsilon": epsilon,
                "sign_consensus_algorithm": ties,
            }
        }
        return (method_def,)


class SLERPMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "interpolation factor. At t=0 will return base_model, at t=1 will return the other one.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Spherical Interpolation Methods"
    DESCRIPTION = """
    Concept: Performs Spherical Linear Interpolation in the weight space between two models. This creates a path 
    along a hypersphere, ensuring the interpolated model maintains a similar "norm" or "magnitude" to the original models.

    Use Cases:

    Creating smooth transitions or intermediate points between two distinct models
    Exploring the space between two models with potentially different capabilities
    Inputs: Requires exactly 2 models. First model is the base_model.

    Key Parameters:

    t (global): Interpolation factor. t=0 yields the base_model, t=1 yields the other model
    """

    def get_method(self, t: float = 1., ):
        method_def = {
            "name": "slerp",
            "settings": {
                "t": t,
            }
        }
        return (method_def,)


class NuSlerpMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nuslerp_flatten": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Set to false to do row-wise/column-wise interpolation instead of treating tensors as vectors.",
                }),
                "nuslerp_row_wise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "SLERP row vectors instead of column vectors.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Spherical Interpolation Methods"
    DESCRIPTION = """
    Concept: An enhanced version of SLERP offering more flexible configuration and faster execution. 
    It allows SLERP between two models directly. If a base_model is provided, NuSLERP calculates task vectors 
    (the difference between each of the two main models and the base model) and then performs SLERP on these 
    task vectors before adding the result back to the base_model.

    Use Cases:
    
    Similar to SLERP, but with more control over weighting for the two primary models.
    To replicate the behavior of the original slerp method (if base_model is not used, and weights are set to 1-t 
    for the first model and t for the second).
    To perform SLERP on task vectors when a base_model is provided, allowing for interpolation of model changes 
    relative to a common ancestor.
    Inputs: Requires exactly 2 models.
    
    Key Parameters:    
    - weight (per-model): Relative weighting for each of the two main models. These are used to calculate the 
    interpolation factor t (where t = model2_weight / (model1_weight + model2_weight)).
    - nuslerp_flatten (global): If false, performs row/column-wise interpolation. Default true
    - nuslerp_row_wise (global): If true (and nuslerp_flatten is false), SLERPs row vectors instead of column vectors. 
    Default false
    """

    def get_method(self, nuslerp_flatten: bool = True, nuslerp_row_wise: bool = False):
        method_def = {
            "name": "nuslerp",
            "settings": {
                "nuslerp_flatten": nuslerp_flatten,
                "nuslerp_row_wise": nuslerp_row_wise,
            }
        }
        return (method_def,)


class KArcherMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_iter": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum iterations for the KArcher mean algorithm. Default 10",
                }),
                "tol": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "tooltip": "Convergence tolerance for the KArcher mean algorithm. Default 1e-5",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Spherical Interpolation Methods"
    DESCRIPTION = """
    KArcher
    Concept: Computes the Karcher mean (also known as the Riemannian barycenter or Fréchet mean) 
    of the input model parameters. This provides a geometrically sound way to average points on a manifold, 
    which is suitable for model weights.

    Use Cases:
    Finding a "central" or "average" model among a set of diverse models in a way that respects the geometry of the weight space
    More robust averaging than simple linear averaging, especially for models far apart in weight space
    Inputs: Takes 2 or more models. No base_model is used.

    Key Parameters:

    max_iter (global): Maximum iterations for the Karcher mean algorithm. Default 10
    tol (global): Convergence tolerance. Default 0.5
    """

    def get_method(self, max_iter: int = 10, tol: float = 0.5):
        method_def = {
            "name": "karcher",
            "settings": {
                "max_iter": max_iter,
                "tol": tol,
            }
        }
        return (method_def,)


class NearSwapMergeMethod:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "similarity_threshold": ("FLOAT", {
                    "default": 0.001,
                    "min": 0,
                    "max": 1,
                    "step": 0.0001,
                    "tooltip": "Similarity threshold for NearSwap merge.",
                }),
            },
        }

    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge/Specialized Methods"
    DESCRIPTION = """
    Nearswap (nearswap)
    Concept: Interpolates the base model with parameters from a secondary model primarily where 
    they are already similar.  The interpolation strength towards the secondary model is inversely proportional 
    to the absolute difference of their parameters, modulated by the t parameter. When the parameters are similar, 
    the interpolation is stronger, and when they are different, it is weaker.

    Use Cases:
    Selectively pulling in similar parameters from a secondary model while preserving different parameters from the 
    base model
    Fine-grained parameter-wise merging that respects the existing structure of the base model
    Inputs: Requires exactly 2 models. One model must be specified as base_model.

    Key Parameters:

    t (global): Controls the interpolation strength. Higher values increase the influence of the secondary model 
    for similar parameters

    Algorithm: For each parameter, computes weight = (t / |base - secondary|).clamp(0, 1), 
    then returns weight * secondary + (1 - weight) * base
    """

    def get_method(self, similarity_threshold: float = 0.001):
        method_def = {
            "name": "nearswap",
            "settings": {
                "similarity_threshold": similarity_threshold,
            }
        }
        return (method_def,)


