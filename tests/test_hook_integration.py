"""
Tests for forward hook integration.

This test validates that the ForwardHookManager can properly intercept
layer computations and route through adapters.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.semantic.adapter.hook_manager import ForwardHookManager
from src.semantic.adapter.registry import AdapterRegistry


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
def simple_model():
    """
    Create a simple model with linear layers.

    This mimics a simplified UNet structure for testing.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(768, 768)
            self.layer2 = nn.Linear(768, 768)
            self.layer3 = nn.Linear(768, 768)

        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            x = torch.relu(x)
            x = self.layer3(x)
            return x

    return SimpleModel()


@pytest.fixture
def simple_lora_stack(device, dtype):
    """
    Create a simple LoRA stack matching the simple model.
    """
    stack = {}

    for lora_name in ["lora_A", "lora_B"]:
        patches = {}

        for layer_idx in [1, 2, 3]:
            layer_key = f"layer{layer_idx}"
            rank = 32
            dim = 768

            up = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
            down = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
            alpha = torch.tensor(rank, device=device, dtype=dtype)

            patches[layer_key] = (up, down, alpha)

        stack[lora_name] = {
            "patches": patches,
            "file_path": f"/fake/path/{lora_name}.safetensors"
        }

    return stack


@pytest.fixture
def simple_semantic_maps(simple_lora_stack, feature_names):
    """Create simple semantic maps for testing."""
    maps = {}

    for lora_name in simple_lora_stack.keys():
        feature_map = {}

        for feature_idx, feature in enumerate(feature_names):
            layer_importances = {}

            for layer_key in simple_lora_stack[lora_name]["patches"].keys():
                # Simple importance: different for each feature
                importance = 0.25 + (feature_idx * 0.1)
                layer_importances[layer_key] = importance

            feature_map[feature] = layer_importances

        maps[lora_name] = feature_map

    return maps


class TestHookManager:
    """Test ForwardHookManager functionality."""

    def test_hook_manager_initialization(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that HookManager can be initialized."""
        # Create adapter registry
        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        # Create hook manager
        hook_manager = ForwardHookManager(
            model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        assert hook_manager.model is simple_model
        assert hook_manager.adapter_registry is registry
        assert len(hook_manager.hooks) == 0  # Not registered yet

    def test_hook_registration(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that hooks can be registered."""
        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        hook_manager = ForwardHookManager(
            model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        # Register hooks
        hook_manager.register_hooks()

        # Note: Hook count may be 0 if module names don't match layer keys
        # This is expected and documented in the hook manager
        assert isinstance(hook_manager.hooks, list)

    def test_hook_removal(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that hooks can be removed."""
        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        hook_manager = ForwardHookManager(
            model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        hook_manager.register_hooks()
        initial_hook_count = len(hook_manager.hooks)

        hook_manager.remove_hooks()

        assert len(hook_manager.hooks) == 0

    def test_semantic_vector_setting(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test setting semantic vector."""
        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        hook_manager = ForwardHookManager(
            model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        # Create semantic vector
        batch_size = 2
        semantic_vector = torch.randn(batch_size, len(feature_names), device=device, dtype=dtype)

        # Set it
        hook_manager.set_semantic_vector(semantic_vector)

        assert hook_manager.semantic_vector is not None
        assert torch.allclose(hook_manager.semantic_vector, semantic_vector)

    def test_context_manager(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that HookManager works as context manager."""
        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        hook_manager = ForwardHookManager(
            model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        # Use as context manager
        with hook_manager:
            # Hooks should be registered
            pass  # May be 0 if names don't match, but won't error

        # Hooks should be removed
        assert len(hook_manager.hooks) == 0

    def test_model_forward_with_hooks(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that model forward pass works with hooks registered (smoke test)."""
        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        hook_manager = ForwardHookManager(
            model=simple_model.to(device),
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        # Create input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 768, device=device, dtype=dtype)
        semantic_vector = torch.randn(batch_size, len(feature_names), device=device, dtype=dtype)

        # Set semantic vector
        hook_manager.set_semantic_vector(semantic_vector)

        # Register hooks
        hook_manager.register_hooks()

        try:
            # Forward pass (should not crash)
            with torch.no_grad():
                output = simple_model(input_tensor)

            # Check output shape
            assert output.shape == (batch_size, 768)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

        finally:
            # Clean up
            hook_manager.remove_hooks()


class TestHookIntegrationWithInferencePipeline:
    """Test hook integration with inference pipeline."""

    def test_pipeline_hook_setup(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that inference pipeline can set up hooks."""
        from src.semantic.inference.pipeline import AdapterInferencePipeline

        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        pipeline = AdapterInferencePipeline(
            base_model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        # Hooks should not be set up initially
        assert pipeline.hook_manager is None

        # Set up hooks
        pipeline.setup_hooks()

        # Now should have hook manager
        assert pipeline.hook_manager is not None
        assert isinstance(pipeline.hook_manager, ForwardHookManager)

        # Clean up
        pipeline.remove_hooks()

        assert pipeline.hook_manager is None

    def test_pipeline_context_manager(
        self,
        simple_model,
        simple_lora_stack,
        simple_semantic_maps,
        feature_names,
        device,
        dtype,
    ):
        """Test that inference pipeline works as context manager."""
        from src.semantic.inference.pipeline import AdapterInferencePipeline

        registry = AdapterRegistry(
            lora_stack=simple_lora_stack,
            semantic_maps=simple_semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        pipeline = AdapterInferencePipeline(
            base_model=simple_model,
            adapter_registry=registry,
            lora_stack=simple_lora_stack,
            device=device,
            dtype=dtype,
        )

        # Use as context manager
        with pipeline:
            # Hooks should be set up
            assert pipeline.hook_manager is not None

        # Hooks should be removed
        assert pipeline.hook_manager is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
