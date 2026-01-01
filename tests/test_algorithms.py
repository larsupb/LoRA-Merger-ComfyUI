"""
Unit tests for merge algorithms.

Tests individual merge algorithm functions from src/merge/algorithms.py.
Mock mergekit dependencies to test algorithm logic in isolation.
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from merge.algorithms import (
    linear_merge,
    get_merge_algorithm,
    MERGE_ALGORITHMS,
)
from merge.utils import apply_weights_to_tensors


class TestApplyWeightsToTensors:
    """Tests for the apply_weights_to_tensors utility function."""

    def test_basic_weighting(self):
        """Test that weights are applied correctly to tensors."""
        tensors = {
            "lora1": torch.ones(10, 10),
            "lora2": torch.ones(10, 10) * 2,
        }
        tensor_parameters = {
            "lora1": {"weight": 0.5},
            "lora2": {"weight": 0.8},
        }

        result = apply_weights_to_tensors(tensors, tensor_parameters)

        assert "lora1" in result
        assert "lora2" in result
        assert torch.allclose(result["lora1"], torch.ones(10, 10) * 0.5)
        assert torch.allclose(result["lora2"], torch.ones(10, 10) * 1.6)

    def test_zero_weight(self):
        """Test that zero weight produces zero tensor."""
        tensors = {"lora1": torch.ones(5, 5)}
        tensor_parameters = {"lora1": {"weight": 0.0}}

        result = apply_weights_to_tensors(tensors, tensor_parameters)

        assert torch.allclose(result["lora1"], torch.zeros(5, 5))

    def test_preserves_tensor_shape(self):
        """Test that tensor shapes are preserved."""
        tensors = {
            "lora1": torch.randn(10, 20),
            "lora2": torch.randn(5, 15, 3),
        }
        tensor_parameters = {
            "lora1": {"weight": 0.7},
            "lora2": {"weight": 0.3},
        }

        result = apply_weights_to_tensors(tensors, tensor_parameters)

        assert result["lora1"].shape == (10, 20)
        assert result["lora2"].shape == (5, 15, 3)


class TestAlgorithmRegistry:
    """Tests for the algorithm registry and dispatcher."""

    def test_all_algorithms_registered(self):
        """Test that all expected algorithms are in the registry."""
        expected_algorithms = [
            "linear",
            "generalized_task_arithmetic",
            "sce",
            "karcher",
            "slerp",
            "nuslerp",
            "nearswap",
        ]

        for alg in expected_algorithms:
            assert alg in MERGE_ALGORITHMS, f"{alg} not in registry"

    def test_get_merge_algorithm_valid(self):
        """Test getting valid algorithm from registry."""
        alg = get_merge_algorithm("linear")
        assert callable(alg)
        assert alg == MERGE_ALGORITHMS["linear"]

    def test_get_merge_algorithm_invalid(self):
        """Test that invalid algorithm name raises error."""
        with pytest.raises(ValueError, match="Unknown merge algorithm"):
            get_merge_algorithm("nonexistent_algorithm")

    def test_algorithm_signature(self):
        """Test that all algorithms have the expected signature."""
        # All merge algorithms should accept these parameters
        expected_params = ["tensors", "gather_tensors", "weight_info", "tensor_parameters", "method_args"]

        for name, func in MERGE_ALGORITHMS.items():
            # Check function has the right parameter names
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            for expected in expected_params:
                assert expected in param_names, f"{name} missing parameter {expected}"


class TestLinearMerge:
    """Tests for linear merge algorithm."""

    @patch('merge.algorithms.LinearMergeTask')
    def test_linear_merge_calls_task(self, mock_task_class):
        """Test that linear merge creates and executes LinearMergeTask."""
        # Setup mocks
        mock_task = Mock()
        mock_task.execute.return_value = torch.ones(10, 10)
        mock_task_class.return_value = mock_task

        mock_tensors = {"lora1": torch.randn(10, 10)}
        mock_gather = Mock()
        mock_weight_info = Mock()
        mock_params = Mock()
        method_args = {"normalize": True}

        # Execute
        result = linear_merge(
            tensors=mock_tensors,
            gather_tensors=mock_gather,
            weight_info=mock_weight_info,
            tensor_parameters=mock_params,
            method_args=method_args
        )

        # Verify task was created with correct args
        mock_task_class.assert_called_once_with(
            gather_tensors=mock_gather,
            tensor_parameters=mock_params,
            normalize=True,
            weight_info=mock_weight_info,
        )

        # Verify task was executed
        mock_task.execute.assert_called_once_with(tensors=mock_tensors)

        # Verify result
        assert torch.allclose(result, torch.ones(10, 10))

    @patch('merge.algorithms.LinearMergeTask')
    def test_linear_merge_default_normalize(self, mock_task_class):
        """Test that normalize defaults to False if not in method_args."""
        mock_task = Mock()
        mock_task.execute.return_value = torch.zeros(5, 5)
        mock_task_class.return_value = mock_task

        # No normalize in method_args
        linear_merge(
            tensors={},
            gather_tensors=Mock(),
            weight_info=Mock(),
            tensor_parameters=Mock(),
            method_args={}
        )

        # Should use default False
        call_kwargs = mock_task_class.call_args.kwargs
        assert call_kwargs["normalize"] == False


# Fixtures for common test data

@pytest.fixture
def sample_tensors():
    """Fixture providing sample tensors for testing."""
    return {
        "lora_1": torch.randn(10, 5),
        "lora_2": torch.randn(10, 5),
    }


@pytest.fixture
def sample_tensor_parameters():
    """Fixture providing sample tensor parameters."""
    return {
        "lora_1": {"weight": 0.6},
        "lora_2": {"weight": 0.4},
    }


@pytest.fixture
def mock_mergekit_objects():
    """Fixture providing mocked mergekit objects."""
    return {
        "gather_tensors": Mock(),
        "weight_info": Mock(name="test.layer"),
        "method_args": {},
    }


class TestIntegrationWithFixtures:
    """Integration tests using fixtures."""

    def test_apply_weights_with_sample_data(self, sample_tensors, sample_tensor_parameters):
        """Test apply_weights with realistic sample data."""
        result = apply_weights_to_tensors(sample_tensors, sample_tensor_parameters)

        # Check all tensors are weighted
        assert len(result) == 2
        assert result["lora_1"].shape == (10, 5)
        assert result["lora_2"].shape == (10, 5)

        # Check weights were applied (result should be scaled versions)
        assert not torch.allclose(result["lora_1"], sample_tensors["lora_1"])
        assert not torch.allclose(result["lora_2"], sample_tensors["lora_2"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
