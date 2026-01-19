"""
Training infrastructure for semantic adapters.

Provides loss functions, curriculum scheduling, data loading, delta providers,
and training loop for learning optimal LoRA composition.
"""

from .losses import SemanticAdapterLoss
from .curriculum import TrainingCurriculum, CurriculumStage
from .trainer import AdapterTrainer
from .data_loader import (
    PromptDataset,
    SemanticBatchSampler,
    HybridSemanticDataLoader,
)
from .delta_provider import (
    DeltaProvider,
    SyntheticDeltaProvider,
    UNetDeltaProvider,
    create_delta_provider,
)

__all__ = [
    "SemanticAdapterLoss",
    "TrainingCurriculum",
    "CurriculumStage",
    "AdapterTrainer",
    "PromptDataset",
    "SemanticBatchSampler",
    "HybridSemanticDataLoader",
    "DeltaProvider",
    "SyntheticDeltaProvider",
    "UNetDeltaProvider",
    "create_delta_provider",
]
