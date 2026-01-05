from typing import Tuple, Dict, Any, List

import torch
import folder_paths

from src.analysis import get_cache, GradientSemanticAnalyzer
from src.nodes_semantic_merge import logger
from src.types import LORA_STACK
from src.utils import detect_lora_architecture


class PMLoRASemanticAnalyzerHeuristic:
    """
    Analyze LoRA stack to build semantic importance maps using heuristic methods.

    This node creates semantic maps that assign importance scores to
    each layer for each feature based on:
    - Layer depth (early/mid/late correlates with feature types)
    - Layer type (attention vs MLP)
    - Architecture-specific patterns

    Supports both predefined features (hair, eyes, clothing) and custom
    feature names (uniform, weapon, background, etc.).

    This is faster than gradient-based analysis and works as a good baseline.
    Can be cached and reused for multiple merges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LoRAStack",),
                "features": (
                    "STRING",
                    {
                        "default": "hair,eyes,face,clothing,accessories,body",
                        "multiline": False,
                    },
                ),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "architecture_hint": (
                    ["auto", "sd15", "sdxl", "dit", "flux", "wan", "qwen"],
                    {"default": "auto"},
                ),
            },
        }

    RETURN_TYPES = ("SemanticMaps", "STRING")
    RETURN_NAMES = ("semantic_maps", "analysis_info")
    FUNCTION = "analyze"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = "Analyze LoRA stack using heuristics to create semantic feature importance maps"

    def analyze(
        self,
        lora_stack: LORA_STACK,
        features: str,
        use_cache: bool = True,
        architecture_hint: str = "auto",
    ) -> Tuple[Dict[str, Any], str]:
        """
        Analyze LoRA stack and create semantic importance maps.

        Args:
            lora_stack: LoRAStack containing multiple LoRAs
            features: Comma-separated feature names
            use_cache: Whether to use cached analysis
            architecture_hint: Architecture hint for better heuristics

        Returns:
            (semantic_maps_dict, analysis_info)
            semantic_maps_dict format: {lora_name: {feature: {layer_key: importance_tensor}}}
        """
        if not lora_stack:
            raise ValueError("LoRA stack is empty")

        # Parse features
        feature_list = [f.strip() for f in features.split(",")]

        # Analyze each LoRA in the stack
        all_semantic_maps = {}
        analysis_results = []
        cache = get_cache()

        for lora_name, lora_entry in lora_stack.items():
            lora_patches = lora_entry.get("patches", {})
            if not lora_patches:
                logger.warning(f"Skipping {lora_name}: no patches")
                continue

            # Check cache
            config = {
                "features": sorted(feature_list),
                "method": "heuristic",
                "architecture": architecture_hint,
            }

            semantic_map = None
            if use_cache:
                cached_map = cache.get(lora_name, lora_patches, config)
                if cached_map is not None:
                    semantic_map = cached_map
                    analysis_results.append(f"{lora_name}: loaded from cache")
                    logger.info(f"Loaded cached semantic map for {lora_name}")

            # Analyze if not cached
            if semantic_map is None:
                # Detect architecture
                if architecture_hint == "auto":
                    arch_name, arch_metadata = detect_lora_architecture(lora_patches)
                else:
                    arch_name = architecture_hint
                    arch_metadata = {}

                # Analyze using heuristics
                semantic_map = self._analyze_heuristic(
                    lora_patches, feature_list, arch_name, arch_metadata
                )

                # Cache results
                if use_cache:
                    metadata = {
                        "lora_name": lora_name,
                        "architecture": arch_name,
                        "num_layers": len(lora_patches),
                        "features": feature_list,
                    }
                    cache.set(lora_name, lora_patches, config, semantic_map, metadata)

                analysis_results.append(
                    f"{lora_name}: analyzed {len(lora_patches)} layers ({arch_name})"
                )
                logger.info(f"Analyzed {lora_name} ({arch_name})")

            # Store in results
            all_semantic_maps[lora_name] = semantic_map

        # Create summary info
        info = (
            f"Analyzed {len(all_semantic_maps)} LoRAs with {len(feature_list)} features:\n"
            + "\n".join(f"  - {result}" for result in analysis_results)
        )

        return (all_semantic_maps, info)

    def _analyze_heuristic(
        self,
        lora_patches: Dict[str, Any],
        features: List[str],
        architecture: str,
        arch_metadata: Dict[str, Any],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Create semantic importance map using heuristics.

        Strategy:
        1. Identify layer depth (early/mid/late)
        2. Assign feature importance based on known correlations:
           - Early layers (0-33%): composition, body, pose
           - Mid layers (33-66%): face, hair, eyes
           - Late layers (66-100%): clothing, accessories, fine details
        3. Weight by layer type (attention layers get higher scores for visible features)
        """
        # Get sorted layer keys
        # Handle mixed key types (tuples and strings) in ComfyUI
        layer_keys = sorted(lora_patches.keys(), key=str)
        num_layers = len(layer_keys)

        # Initialize semantic map
        semantic_map: Dict[str, Dict[str, torch.Tensor]] = {
            feature: {} for feature in features
        }

        # Process each layer
        for idx, layer_key in enumerate(layer_keys):
            # Compute relative depth (0.0 to 1.0)
            depth = idx / max(num_layers - 1, 1)

            # Get layer patches
            layer_patches = lora_patches[layer_key]
            if not isinstance(layer_patches, dict):
                continue

            # Compute importance tensor (average over all sublayers)
            importance_tensors = []
            for sublayer_key, tensor in layer_patches.items():
                if isinstance(tensor, torch.Tensor):
                    # Create importance scores (initially uniform)
                    importance = torch.ones_like(tensor, dtype=torch.float32)
                    importance_tensors.append(importance)

            if not importance_tensors:
                continue

            # Average importance across sublayers
            avg_importance = torch.stack(importance_tensors).mean(dim=0)

            # Assign feature-specific importance based on depth and type
            layer_type = self._classify_layer_type(layer_key)

            for feature in features:
                # Get feature-specific multiplier based on depth
                multiplier = self._get_feature_depth_multiplier(
                    feature, depth, layer_type, architecture
                )

                # Store scaled importance
                semantic_map[feature][layer_key] = avg_importance * multiplier

        # Normalize across features for each layer
        semantic_map = self._normalize_feature_importance(semantic_map)

        return semantic_map

    def _classify_layer_type(self, layer_key) -> str:
        """Classify layer as attention, mlp, or other."""
        # Convert to string if it's a tuple (ComfyUI internal format)
        key_str = str(layer_key) if isinstance(layer_key, tuple) else layer_key
        key_lower = key_str.lower()
        if any(x in key_lower for x in ["attn", "attention"]):
            return "attention"
        elif any(x in key_lower for x in ["mlp", "ff", "feed_forward", "ffn"]):
            return "mlp"
        else:
            return "other"

    def _get_feature_depth_multiplier(
        self,
        feature: str,
        depth: float,
        layer_type: str,
        architecture: str,
    ) -> float:
        """
        Get importance multiplier for a feature at a given layer depth.

        Supports both predefined features (with known depth preferences) and
        custom features (assigned to mid-range depth).

        Args:
            feature: Feature name (hair, eyes, uniform, etc.)
            depth: Relative layer depth (0.0 = early, 1.0 = late)
            layer_type: Layer type (attention, mlp, other)
            architecture: Model architecture

        Returns:
            Multiplier (0.0 to 2.0, typically)
        """
        # Feature depth preferences (peak importance)
        # Based on empirical observations of diffusion model layer functions
        feature_depth_prefs = {
            # Early layers: overall structure
            "body": 0.2,
            "composition": 0.2,
            "pose": 0.2,
            # Mid-early layers: face structure
            "face": 0.4,
            # Mid layers: hair, major features
            "hair": 0.5,
            "eyes": 0.5,
            # Mid-late layers: clothing, style
            "clothing": 0.7,
            "outfit": 0.7,
            "uniform": 0.7,  # Common custom feature
            # Late layers: fine details
            "accessories": 0.8,
            "details": 0.9,
            "texture": 0.9,
        }

        # Default to mid-range for unknown features
        # Custom features (e.g., "weapon", "background", "tattoo") get 0.5
        preferred_depth = feature_depth_prefs.get(feature, 0.5)

        # Log if using default for custom feature
        if feature not in feature_depth_prefs:
            logger.debug(
                f"Custom feature '{feature}' not in predefined list, "
                f"using default depth preference {preferred_depth}"
            )

        # Gaussian-like distribution centered at preferred depth
        # Width of 0.3 means feature importance spans about 30% of depth range
        width = 0.3
        distance = abs(depth - preferred_depth)
        gaussian_weight = 2.0 * torch.exp(
            torch.tensor(-((distance / width) ** 2))
        ).item()

        # Attention layers get bonus for visible features
        if layer_type == "attention" and feature in [
            "hair",
            "eyes",
            "face",
            "clothing",
        ]:
            gaussian_weight *= 1.2

        # MLP layers get bonus for fine details
        if layer_type == "mlp" and feature in ["accessories", "details", "texture"]:
            gaussian_weight *= 1.2

        # Ensure minimum importance (even "wrong" layers contribute a bit)
        return max(0.2, gaussian_weight)

    def _normalize_feature_importance(
        self, semantic_map: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Normalize importance scores so features sum to 1.0 for each layer.

        This ensures that feature weights can be directly used as mixing weights.
        """
        # Collect all layer keys
        all_layers = set()
        for feature_map in semantic_map.values():
            all_layers.update(feature_map.keys())

        # Normalize per layer
        for layer_key in all_layers:
            # Collect tensors for this layer
            layer_tensors = {}
            for feature, feature_map in semantic_map.items():
                if layer_key in feature_map:
                    layer_tensors[feature] = feature_map[layer_key]

            if not layer_tensors:
                continue

            # Stack and sum
            stacked = torch.stack(list(layer_tensors.values()))
            total = stacked.sum(dim=0, keepdim=True)

            # Avoid division by zero
            total = torch.clamp(total, min=1e-8)

            # Normalize
            for feature, tensor in layer_tensors.items():
                semantic_map[feature][layer_key] = tensor / total.squeeze(0)

        return semantic_map


class PMLoRASemanticAnalyzerGradient:
    """
    Analyze LoRA stack using gradient-based attribution (diffusers-based).

    This node uses the diffusers library to load models independently from ComfyUI,
    enabling proper gradient computation for accurate feature attribution. More
    accurate than heuristics but requires GPU and takes longer (~10-30 seconds per LoRA).

    **How it works:**
    1. Loads model architecture from HuggingFace/diffusers (cached after first run)
    2. Loads custom checkpoint weights from your .safetensors file
    3. Applies LoRA patches by merging into weights
    4. Runs forward passes with feature-specific prompts
    5. Computes gradients to identify important weights
    6. Builds semantic importance maps

    **Currently Supported Architectures:**
    - zImage: Fully implemented
    - SDXL: Fully implemented (UNet-based, works with 8GB VRAM with offloading)
    - SD1.5: Fully implemented (UNet-based, works with 8GB VRAM with offloading)
    - Flux, Qwen, Wan: Coming soon

    **Requirements:**
    - checkpoint_name: Select checkpoint from dropdown (searches checkpoints/ and diffusion_models/ folders)
    - CLIP: For encoding feature prompts
    - diffusers library (auto-installed from requirements.txt)

    Supports both predefined features (hair, eyes, clothing) and custom
    feature names (uniform, weapon, background, etc.). Custom features
    automatically get generated prompts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available checkpoints from both folders
        checkpoint_files = []

        # Add files from checkpoints folder
        checkpoints = folder_paths.get_filename_list("checkpoints")
        checkpoint_files.extend([f"checkpoints/{f}" for f in checkpoints])

        # Add files from diffusion_models folder
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        checkpoint_files.extend([f"diffusion_models/{f}" for f in diffusion_models])

        # Add empty option for when MODEL is connected
        checkpoint_files.insert(0, "")

        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "lora_stack": ("LoRAStack",),
                "checkpoint_name": (
                    checkpoint_files,
                    {
                        "default": "",
                        "tooltip": "Select checkpoint from dropdown. Leave empty if MODEL from 'PM Checkpoint Loader (with Path)' is connected"
                    }
                ),
                "features": (
                    "STRING",
                    {
                        "default": "hair,eyes,face,clothing,accessories,body",
                        "multiline": False,
                    },
                ),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "architecture_hint": (
                    ["auto", "sd15", "sdxl", "dit", "flux", "wan", "qwen", "zimage"],
                    {"default": "auto"},
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto", "tooltip": "Device for gradient computation"}
                ),
                "dtype": (
                    ["auto", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"],
                    {"default": "auto", "tooltip": "Data type for computation (FP8 for memory-efficient loading)"}
                ),
                "num_samples": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "tooltip": "Number of prompts per feature (more = slower but more stable)"
                    }
                ),
                "offload_layers": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable layer offloading to CPU to reduce VRAM usage (requires accelerate library)"
                    }
                ),
                "gpu_memory_gb": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 96.0,
                        "step": 0.5,
                        "tooltip": "GPU memory limit in GB (0 = auto-detect 80% of VRAM). Lower values = more layers offloaded. Examples: 2.0 (very aggressive), 4.0 (moderate), 8.0 (light). Only used if offload_layers=True"
                    }
                ),
            },
        }

    RETURN_TYPES = ("SemanticMaps", "STRING")
    RETURN_NAMES = ("semantic_maps", "analysis_info")
    FUNCTION = "analyze"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = "Analyze LoRA stack using gradient-based attribution for accurate semantic feature maps"

    def analyze(
        self,
        clip: Any,
        lora_stack: LORA_STACK,
        checkpoint_name: str,
        features: str,
        use_cache: bool = True,
        architecture_hint: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        num_samples: int = 3,
        offload_layers: bool = False,
        gpu_memory_gb: float = 0.0,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Analyze LoRA stack using gradient-based attribution.

        Args:
            lora_stack: LoRAStack containing multiple LoRAs
            clip: ComfyUI CLIP for text encoding
            features: Comma-separated feature names
            use_cache: Whether to use cached analysis
            model: Optional MODEL (auto-extracts checkpoint_path if from 'PM Checkpoint Loader (with Path)')
            checkpoint_name: Checkpoint filename from dropdown. Optional if model has checkpoint_path attribute
            architecture_hint: Architecture hint ("auto", "sd15", "sdxl", "zimage", etc.)
            device: Device for computation ("auto", "cuda", "cpu")
            dtype: Data type for computation ("auto", "float32", "float16", "bfloat16")
            num_samples: Number of prompts per feature

        Returns:
            (semantic_maps_dict, analysis_info)
            semantic_maps_dict format: {lora_file_path: {feature: {layer_key: importance_tensor}}}
        """
        if not lora_stack:
            raise ValueError("LoRA stack is empty")

        # Determine checkpoint path
        checkpoint_path = None

        # If not from MODEL, use checkpoint_name from dropdown
        if not checkpoint_path and checkpoint_name and checkpoint_name.strip() != "":
            # Parse folder prefix (e.g., "checkpoints/model.safetensors" or "diffusion_models/model.safetensors")
            if "/" in checkpoint_name:
                folder_type, filename = checkpoint_name.split("/", 1)
                checkpoint_path = folder_paths.get_full_path(folder_type, filename)
                logger.info(f"Using checkpoint from dropdown: {checkpoint_path}")
            else:
                raise ValueError(f"Invalid checkpoint_name format: {checkpoint_name}")

        # Validate we have a checkpoint path
        if not checkpoint_path:
            raise ValueError(
                "checkpoint_path is required for gradient analysis. Either:\n"
                "1. Select a checkpoint from the dropdown, OR\n"
                "2. Use 'PM Checkpoint Loader (with Path)' node and connect MODEL input"
            )

        # Parse features
        feature_list = [f.strip() for f in features.split(",")]

        # Determine device and dtype
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        if dtype == "auto":
            dtype_obj = torch.float32
        else:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float8_e4m3fn": torch.float8_e4m3fn,
                "float8_e5m2": torch.float8_e5m2,
            }
            dtype_obj = dtype_map.get(dtype, torch.float32)

        logger.info(f"Gradient analysis using device={device_obj}, dtype={dtype_obj}")

        # Create gradient analyzer
        analyzer = GradientSemanticAnalyzer(clip, device_obj, dtype_obj)

        # Analyze each LoRA in the stack
        all_semantic_maps = {}
        analysis_results = []
        cache = get_cache()

        for lora_name, stack_entry in lora_stack.items():
            # Check cache
            config = {
                "features": sorted(feature_list),
                "method": "gradient",
                "architecture": architecture_hint,
                "num_samples": num_samples,
                "checkpoint_path": checkpoint_path,
            }
            lora_file_path = stack_entry["file_path"]

            semantic_map = None
            if use_cache:
                cached_map = cache.get(lora_file_path, config)
                if cached_map is not None:
                    semantic_map = cached_map
                    analysis_results.append(f"{lora_file_path}: loaded from cache")
                    logger.info(f"Loaded cached gradient semantic map for {lora_file_path}")

            # Analyze if not cached
            if semantic_map is None:
                # Detect architecture if auto
                arch_name = architecture_hint

                logger.info(f"Running gradient analysis for {lora_file_path} ({arch_name})...")

                # Run gradient-based analysis with checkpoint_path
                # Convert gpu_memory_gb to None if 0.0 (auto-detect)
                gpu_mem = None if gpu_memory_gb == 0.0 else gpu_memory_gb

                semantic_map = analyzer.analyze_lora(
                    lora_path=lora_file_path,
                    features=feature_list,
                    architecture=arch_name,
                    checkpoint_path=checkpoint_path,
                    num_samples=num_samples,
                    offload_layers=offload_layers,
                    gpu_memory_gb=gpu_mem,
                )

                # Cache results
                if use_cache:
                    metadata = {
                        "lora_file_path": lora_file_path,
                        "architecture": arch_name,
                        "features": feature_list,
                        "num_samples": num_samples,
                    }
                    cache.set(lora_file_path, config, semantic_map, metadata)

                analysis_results.append(
                    f"{lora_file_path}: analyzed using gradients ({arch_name})"
                )
                logger.info(f"Gradient analysis complete for {lora_file_path}")

            # Store in results
            all_semantic_maps[lora_name] = semantic_map

        # Create summary info
        info = (
            f"Gradient-based analysis of {len(all_semantic_maps)} LoRAs with {len(feature_list)} features:\n"
            + f"Checkpoint: {checkpoint_path}\n"
            + f"Device: {device_obj}, dtype: {dtype_obj}\n"
            + "\n".join(f"  - {result}" for result in analysis_results)
        )

        if not all_semantic_maps:
            raise RuntimeError("No LoRAs were successfully analyzed")

        return (all_semantic_maps, info)


# ComfyUI node registration exports
NODE_CLASS_MAPPINGS = {
    "PM LoRA Semantic Analyzer (Heuristic)": PMLoRASemanticAnalyzerHeuristic,
    "PM LoRA Semantic Analyzer (Gradient)": PMLoRASemanticAnalyzerGradient,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PM LoRA Semantic Analyzer (Heuristic)": "PM LoRA Semantic Analyzer (Heuristic)",
    "PM LoRA Semantic Analyzer (Gradient)": "PM LoRA Semantic Analyzer (Gradient)",
}
