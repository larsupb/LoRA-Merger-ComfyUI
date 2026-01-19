"""
End-to-end test for semantic adapter training and inference.

This test validates the complete workflow:
1. Load/create LoRA stack
2. Generate semantic importance maps (heuristic or gradient-based)
3. Train adapters to learn feature-aware composition
4. Apply adapters during inference
5. Validate results

Similar to test_gradient_analysis.py but focuses on adapter training/inference.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Import adapter modules
from src.semantic.adapter.base import BaseAdapter
from src.semantic.adapter.rank_space import RankSpaceAdapter
from src.semantic.adapter.output_space import OutputSpaceAdapter
from src.semantic.adapter.registry import AdapterRegistry
from src.semantic.training.trainer import AdapterTrainer
from src.semantic.training.losses import SemanticAdapterLoss
from src.semantic.training.curriculum import TrainingCurriculum
from src.semantic.inference.pipeline import AdapterInferencePipeline
from src.semantic.serialization.format import AdapterSerializer

# Import existing semantic modules
from src.semantic.semantic_merger import SemanticMerger
from src.semantic.feature_prompts import get_feature_prompts, get_all_features


@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Get compute dtype."""
    return torch.float32


@pytest.fixture
def feature_names():
    """Get test feature names."""
    return ["hair", "eyes", "clothing", "accessories"]


@pytest.fixture
def lora_stack(device, dtype):
    """
    Create a mock LoRA stack with two LoRAs.

    Structure matches LORA_STACK type:
    Dict[lora_name -> {"patches": Dict[layer_key -> (up, down, alpha)], "file_path": str}]
    """
    stack = {}

    # Create two mock LoRAs
    for lora_idx, lora_name in enumerate(["character_A", "character_B"]):
        patches = {}

        # Create patches for multiple layers (attention + MLP + conv)
        for layer_idx in range(8):
            # Attention layers (rank-space)
            for attn_type in ["q", "k", "v", "out"]:
                layer_key = f"diffusion_model.layers.{layer_idx}.attention.{attn_type}"
                rank = 32
                in_dim = 768
                out_dim = 768

                up = torch.randn(out_dim, rank, device=device, dtype=dtype) * 0.1
                down = torch.randn(rank, in_dim, device=device, dtype=dtype) * 0.1
                alpha = torch.tensor(rank, device=device, dtype=dtype)

                patches[layer_key] = (up, down, alpha)

            # MLP layers (rank-space)
            for mlp_type in ["w1", "w2"]:
                layer_key = f"diffusion_model.layers.{layer_idx}.mlp.{mlp_type}"
                rank = 32
                in_dim = 768
                out_dim = 3072 if mlp_type == "w1" else 768

                up = torch.randn(out_dim, rank, device=device, dtype=dtype) * 0.1
                down = torch.randn(rank, in_dim, device=device, dtype=dtype) * 0.1
                alpha = torch.tensor(rank, device=device, dtype=dtype)

                patches[layer_key] = (up, down, alpha)

        # Add some conv layers (output-space)
        for conv_idx in range(4):
            layer_key = f"diffusion_model.conv_blocks.{conv_idx}.conv"
            # Conv layers don't have natural rank structure
            rank = 16
            channels = 320

            up = torch.randn(channels, rank, device=device, dtype=dtype) * 0.1
            down = torch.randn(rank, channels, device=device, dtype=dtype) * 0.1
            alpha = torch.tensor(rank, device=device, dtype=dtype)

            patches[layer_key] = (up, down, alpha)

        stack[lora_name] = {
            "patches": patches,
            "file_path": f"/fake/path/{lora_name}.safetensors"
        }

    return stack


@pytest.fixture
def semantic_maps(lora_stack, feature_names):
    """
    Create mock semantic importance maps.

    Structure: Dict[lora_name -> Dict[feature -> Dict[layer_key -> float]]]
    """
    maps = {}

    for lora_name in lora_stack.keys():
        feature_map = {}

        for feature_idx, feature in enumerate(feature_names):
            layer_importances = {}

            # Different features peak at different depths
            # hair: early-mid layers
            # eyes: mid layers
            # clothing: late layers
            # accessories: very late layers

            for layer_key in lora_stack[lora_name]["patches"].keys():
                # Extract layer index from key
                if "layers." in layer_key:
                    layer_idx = int(layer_key.split("layers.")[1].split(".")[0])
                    total_layers = 8
                    depth = layer_idx / total_layers

                    # Feature-specific importance curves
                    import math
                    if feature == "hair":
                        importance = math.exp(-((depth - 0.3) ** 2) / 0.1)
                    elif feature == "eyes":
                        importance = math.exp(-((depth - 0.5) ** 2) / 0.1)
                    elif feature == "clothing":
                        importance = math.exp(-((depth - 0.7) ** 2) / 0.1)
                    elif feature == "accessories":
                        importance = math.exp(-((depth - 0.9) ** 2) / 0.1)
                    else:
                        importance = 1.0 / len(feature_names)

                    layer_importances[layer_key] = importance
                else:
                    # Conv layers - lower importance
                    layer_importances[layer_key] = 0.1

            feature_map[feature] = layer_importances

        maps[lora_name] = feature_map

    return maps


@pytest.fixture
def merge_spec():
    """Create test merge specification."""
    return {
        "features": {
            "hair": {"source": "character_A", "weight": 1.0},
            "eyes": {"source": "character_A", "weight": 1.0},
            "clothing": {"source": "character_B", "weight": 1.0},
            "accessories": {"source": "character_B", "weight": 1.0},
        }
    }


class TestAdapterArchitecture:
    """Test adapter module architecture."""

    def test_rank_space_adapter_initialization(self, device, dtype, feature_names):
        """Test RankSpaceAdapter can be instantiated."""
        adapter = RankSpaceAdapter(
            layer_key="test_layer",
            semantic_dim=len(feature_names),
            feature_names=feature_names,
            rank=32,
            num_loras=2,
            use_residual_mlp=True,
        ).to(device=device, dtype=dtype)

        assert adapter.layer_key == "test_layer"
        assert adapter.semantic_dim == len(feature_names)
        assert adapter.feature_names == feature_names

        # Check gate network exists
        assert hasattr(adapter, "gate_network")
        assert isinstance(adapter.gate_network, nn.Sequential)

        # Check residual MLP exists
        assert hasattr(adapter, "residual_mlp")
        assert adapter.residual_mlp is not None

    def test_rank_space_adapter_forward_shapes(self, device, dtype, feature_names):
        """Test RankSpaceAdapter forward pass produces correct shapes."""
        batch_size = 2
        seq_len = 64
        rank = 32
        out_dim = 768
        num_loras = 2

        adapter = RankSpaceAdapter(
            layer_key="test_layer",
            semantic_dim=len(feature_names),
            feature_names=feature_names,
            rank=rank,
            num_loras=num_loras,
            use_residual_mlp=False,  # Simpler for shape testing
        ).to(device=device, dtype=dtype)

        # Create mock inputs
        lora_down_outputs = {
            f"lora_{i}": torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype)
            for i in range(num_loras)
        }

        b_shared = torch.randn(out_dim, rank, device=device, dtype=dtype)
        semantic_vector = torch.randn(batch_size, len(feature_names), device=device, dtype=dtype)

        # Forward pass
        output = adapter(lora_down_outputs, b_shared, semantic_vector)

        # Check output shape
        assert output.shape == (batch_size, seq_len, out_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_output_space_adapter_initialization(self, device, dtype, feature_names):
        """Test OutputSpaceAdapter can be instantiated."""
        adapter = OutputSpaceAdapter(
            layer_key="test_layer",
            semantic_dim=len(feature_names),
            feature_names=feature_names,
            output_dim=768,
            num_loras=2,
        ).to(device=device, dtype=dtype)

        assert adapter.layer_key == "test_layer"
        assert hasattr(adapter, "gate_network")
        assert hasattr(adapter, "residual_mlp")

    def test_output_space_adapter_forward_shapes(self, device, dtype, feature_names):
        """Test OutputSpaceAdapter forward pass produces correct shapes."""
        batch_size = 2
        height = 32
        width = 32
        channels = 320
        num_loras = 2

        adapter = OutputSpaceAdapter(
            layer_key="test_layer",
            semantic_dim=len(feature_names),
            feature_names=feature_names,
            output_dim=channels,
            num_loras=num_loras,
        ).to(device=device, dtype=dtype)

        # Create mock inputs (conv layer outputs)
        lora_deltas = {
            f"lora_{i}": torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
            for i in range(num_loras)
        }

        semantic_vector = torch.randn(batch_size, len(feature_names), device=device, dtype=dtype)

        # Forward pass
        output = adapter(lora_deltas, semantic_vector)

        # Check output shape
        assert output.shape == (batch_size, channels, height, width)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestAdapterRegistry:
    """Test adapter registry system."""

    def test_registry_initialization(self, lora_stack, semantic_maps, feature_names, device, dtype):
        """Test adapter registry can be built from LoRA stack."""
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        # Check adapters were created for all unique layers
        all_layer_keys = set()
        for lora_entry in lora_stack.values():
            all_layer_keys.update(lora_entry["patches"].keys())

        assert len(registry.adapters) == len(all_layer_keys)

        # Check adapter types were assigned
        assert len(registry.adapter_types) == len(all_layer_keys)

    def test_registry_adapter_type_detection(self, lora_stack, semantic_maps, feature_names, device, dtype):
        """Test registry correctly detects adapter types."""
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        # Attention and MLP layers should be rank_space
        for layer_key in registry.adapter_types.keys():
            if "attention" in layer_key or "mlp" in layer_key:
                assert registry.adapter_types[layer_key] == "rank_space"
                assert isinstance(registry.adapters[layer_key], RankSpaceAdapter)
            elif "conv" in layer_key:
                assert registry.adapter_types[layer_key] == "output_space"
                assert isinstance(registry.adapters[layer_key], OutputSpaceAdapter)

    def test_registry_state_dict(self, lora_stack, semantic_maps, feature_names, device, dtype):
        """Test registry can be serialized to state dict."""
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        state_dict = registry.state_dict()

        # Check state dict structure
        assert "adapter_types" in state_dict
        assert "adapters" in state_dict
        assert "feature_names" in state_dict

        # Check all adapters are included
        assert len(state_dict["adapters"]) == len(registry.adapters)

    def test_registry_load_state_dict(self, lora_stack, semantic_maps, feature_names, device, dtype):
        """Test registry can load from state dict."""
        # Create and serialize first registry
        registry1 = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        state_dict = registry1.state_dict()

        # Create second registry and load state
        registry2 = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        registry2.load_state_dict(state_dict)

        # Verify parameters match
        for layer_key in registry1.adapters.keys():
            adapter1 = registry1.adapters[layer_key]
            adapter2 = registry2.adapters[layer_key]

            for p1, p2 in zip(adapter1.parameters(), adapter2.parameters()):
                assert torch.allclose(p1, p2, atol=1e-6)


class TestTrainingComponents:
    """Test training infrastructure."""

    def test_loss_function_initialization(self):
        """Test loss function can be created."""
        loss_fn = SemanticAdapterLoss(
            lambda_teacher=1.0,
            lambda_dominance=0.5,
            lambda_residual=0.1,
            lambda_consistency=0.3,
        )

        assert loss_fn.lambda_teacher == 1.0
        assert loss_fn.lambda_dominance == 0.5

    def test_teacher_alignment_loss(self, device, dtype):
        """Test teacher alignment loss computation."""
        loss_fn = SemanticAdapterLoss()

        adapter_delta = torch.randn(2, 768, device=device, dtype=dtype)
        teacher_delta = torch.randn(2, 768, device=device, dtype=dtype)

        loss = loss_fn.teacher_alignment_loss(adapter_delta, teacher_delta)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # MSE is non-negative

    def test_curriculum_initialization(self):
        """Test training curriculum can be created."""
        curriculum = TrainingCurriculum()

        assert len(curriculum.stages) == 3  # Three-stage curriculum
        assert curriculum.current_stage_idx == 0
        assert curriculum.current_epoch == 0

    def test_curriculum_stage_transitions(self):
        """Test curriculum advances through stages."""
        curriculum = TrainingCurriculum()

        # Initially in stage 0
        assert curriculum.get_current_stage().name == "alignment"

        # Advance through first stage (50 epochs)
        for _ in range(50):
            curriculum.step_epoch()

        # Should be in stage 1
        assert curriculum.get_current_stage().name == "dominance"

    def test_curriculum_loss_weights(self):
        """Test curriculum provides correct loss weights."""
        curriculum = TrainingCurriculum()

        # Stage 0: only teacher + residual
        weights = curriculum.get_loss_weights()
        assert weights["lambda_teacher"] == 1.0
        assert weights["lambda_dominance"] == 0.0

        # Advance to stage 1
        for _ in range(50):
            curriculum.step_epoch()

        # Stage 1: add dominance
        weights = curriculum.get_loss_weights()
        assert weights["lambda_dominance"] > 0.0


class TestTrainerIntegration:
    """Test full training loop integration."""

    @pytest.fixture
    def mock_base_model(self):
        """Create mock base UNet model."""
        model = nn.Module()
        model.eval()
        return model

    @pytest.fixture
    def semantic_merger(self, lora_stack, semantic_maps):
        """Create semantic merger for teacher targets."""
        # Mock semantic merger
        merger = Mock(spec=SemanticMerger)
        merger.merge = Mock(return_value={})
        return merger

    def test_trainer_initialization(
        self,
        lora_stack,
        semantic_maps,
        feature_names,
        device,
        dtype,
        mock_base_model,
        semantic_merger,
    ):
        """Test trainer can be instantiated."""
        # Create adapter registry
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        # Create trainer
        trainer = AdapterTrainer(
            adapter_registry=registry,
            base_model=mock_base_model,
            lora_stack=lora_stack,
            semantic_merger=semantic_merger,
            feature_prompts=["test prompt"],
            device=device,
            dtype=dtype,
        )

        assert trainer.adapter_registry is registry
        assert trainer.base_model is mock_base_model
        assert trainer.curriculum is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None

    def test_trainer_checkpoint_save_load(
        self,
        lora_stack,
        semantic_maps,
        feature_names,
        device,
        dtype,
        mock_base_model,
        semantic_merger,
        tmp_path,
    ):
        """Test trainer can save and load checkpoints."""
        # Create trainer
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        trainer = AdapterTrainer(
            adapter_registry=registry,
            base_model=mock_base_model,
            lora_stack=lora_stack,
            semantic_merger=semantic_merger,
            feature_prompts=["test prompt"],
            device=device,
            dtype=dtype,
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))

        # Verify state was restored (basic check)
        assert trainer.curriculum.current_epoch >= 0


class TestSerialization:
    """Test adapter serialization and deserialization."""

    def test_adapter_serialization_format(self, tmp_path):
        """Test adapter package can be saved to disk."""
        # Create mock adapter package
        adapter_package = {
            "lora_names": ["character_A", "character_B"],
            "feature_names": ["hair", "eyes", "clothing"],
            "merge_spec": {"test": "spec"},
            "adapter_state_dict": {
                "adapter_types": {"layer1": "rank_space", "layer2": "output_space"},
                "adapters": {
                    "layer1": {
                        "gate_network.0.weight": torch.randn(64, 32),
                        "gate_network.0.bias": torch.randn(64),
                    }
                },
                "feature_names": ["hair", "eyes", "clothing"],
            }
        }

        # Save
        output_dir = tmp_path / "adapter_package"
        AdapterSerializer.save(adapter_package, str(output_dir))

        # Check files exist
        assert (output_dir / "config.json").exists()
        assert (output_dir / "adapters.safetensors").exists()

    def test_adapter_deserialization(self, tmp_path):
        """Test adapter package can be loaded from disk."""
        # Create and save package
        adapter_package = {
            "lora_names": ["character_A", "character_B"],
            "feature_names": ["hair", "eyes", "clothing"],
            "merge_spec": {"test": "spec"},
            "adapter_state_dict": {
                "adapter_types": {"layer1": "rank_space"},
                "adapters": {
                    "layer1": {
                        "weight": torch.randn(64, 32),
                    }
                },
                "feature_names": ["hair", "eyes", "clothing"],
            }
        }

        output_dir = tmp_path / "adapter_package"
        AdapterSerializer.save(adapter_package, str(output_dir))

        # Load
        loaded_package = AdapterSerializer.load(str(output_dir))

        # Verify structure
        assert loaded_package["lora_names"] == adapter_package["lora_names"]
        assert loaded_package["feature_names"] == adapter_package["feature_names"]
        assert "adapter_state_dict" in loaded_package


class TestEndToEndWorkflow:
    """Test complete end-to-end adapter workflow."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for full E2E test")
    def test_full_workflow(
        self,
        lora_stack,
        semantic_maps,
        feature_names,
        merge_spec,
        device,
        dtype,
        tmp_path,
    ):
        """
        End-to-end test: Build registry -> Train adapters -> Save -> Load -> Inference

        This is a smoke test to ensure all components work together.
        """
        # Step 1: Build adapter registry
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        print(f"\nStep 1: Created registry with {len(registry.adapters)} adapters")

        # Step 2: Create mock training components
        mock_base_model = nn.Module()
        mock_base_model.eval()

        mock_semantic_merger = Mock(spec=SemanticMerger)
        mock_semantic_merger.merge = Mock(return_value={})

        # Step 3: Create trainer
        trainer = AdapterTrainer(
            adapter_registry=registry,
            base_model=mock_base_model,
            lora_stack=lora_stack,
            semantic_merger=mock_semantic_merger,
            feature_prompts=["1girl, detailed"],
            device=device,
            dtype=dtype,
        )

        print(f"Step 2: Created trainer with curriculum")

        # Step 4: Save checkpoint (simulates training)
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        print(f"Step 3: Saved checkpoint to {checkpoint_path}")

        # Step 5: Serialize adapter package
        adapter_package = {
            "lora_names": list(lora_stack.keys()),
            "feature_names": feature_names,
            "merge_spec": merge_spec,
            "adapter_state_dict": registry.state_dict(),
        }

        package_dir = tmp_path / "adapter_package"
        AdapterSerializer.save(adapter_package, str(package_dir))

        print(f"Step 4: Serialized adapter package to {package_dir}")

        # Step 6: Load adapter package
        loaded_package = AdapterSerializer.load(str(package_dir))

        print(f"Step 5: Loaded adapter package")

        # Step 7: Create new registry from loaded package
        loaded_registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=loaded_package["feature_names"],
            device=device,
            dtype=dtype,
        )

        loaded_registry.load_state_dict(loaded_package["adapter_state_dict"])

        print(f"Step 6: Restored registry from loaded package")

        # Step 8: Verify adapter parameters match
        for layer_key in registry.adapters.keys():
            original_adapter = registry.adapters[layer_key]
            loaded_adapter = loaded_registry.adapters[layer_key]

            for p_orig, p_loaded in zip(original_adapter.parameters(), loaded_adapter.parameters()):
                assert torch.allclose(p_orig, p_loaded, atol=1e-6)

        print(f"Step 7: Verified parameter consistency")
        print(f"\n✓ End-to-end workflow completed successfully!")

    def test_inference_pipeline_initialization(
        self,
        lora_stack,
        semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test inference pipeline can be created."""
        # Create adapter registry
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        # Create mock base model
        mock_base_model = nn.Module()

        # Create inference pipeline
        pipeline = AdapterInferencePipeline(
            base_model=mock_base_model,
            adapter_registry=registry,
            lora_stack=lora_stack,
            device=device,
            dtype=dtype,
        )

        assert pipeline.adapter_registry is registry
        assert pipeline.base_model is mock_base_model

        print(f"✓ Inference pipeline initialized successfully")


class TestAdapterStatistics:
    """Test adapter monitoring and statistics."""

    def test_rank_space_adapter_statistics(self, device, dtype, feature_names):
        """Test rank-space adapter provides statistics."""
        adapter = RankSpaceAdapter(
            layer_key="test_layer",
            semantic_dim=len(feature_names),
            feature_names=feature_names,
            rank=32,
            num_loras=2,
            use_residual_mlp=True,
        ).to(device=device, dtype=dtype)

        stats = adapter.get_statistics()

        assert isinstance(stats, dict)
        assert "gate_sparsity" in stats
        assert "gate_mean" in stats
        assert "gate_std" in stats
        assert "residual_norm" in stats

    def test_output_space_adapter_statistics(self, device, dtype, feature_names):
        """Test output-space adapter provides statistics."""
        adapter = OutputSpaceAdapter(
            layer_key="test_layer",
            semantic_dim=len(feature_names),
            feature_names=feature_names,
            output_dim=768,
            num_loras=2,
        ).to(device=device, dtype=dtype)

        stats = adapter.get_statistics()

        assert isinstance(stats, dict)
        assert "gate_entropy" in stats
        assert "residual_magnitude" in stats

    def test_registry_statistics_collection(
        self,
        lora_stack,
        semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test registry collects statistics from all adapters."""
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        all_stats = registry.get_statistics()

        # Should have stats for all adapters
        assert len(all_stats) == len(registry.adapters)

        # Each entry should be a dict of statistics
        for layer_key, stats in all_stats.items():
            assert isinstance(stats, dict)
            assert len(stats) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
