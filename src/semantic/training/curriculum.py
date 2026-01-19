"""
Training curriculum for progressive adapter training.

Implements a 3-stage curriculum:
- Stage 1: Basic alignment with semantic merge
- Stage 2: Add feature dominance constraints
- Stage 3: Full training with output consistency
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CurriculumStage:
    """
    Configuration for a training curriculum stage.

    Attributes:
        name: Stage name (e.g., "alignment", "dominance", "full")
        epochs: Number of epochs for this stage
        lambda_teacher: Weight for teacher alignment loss
        lambda_dominance: Weight for feature dominance loss
        lambda_residual: Weight for residual economy loss
        lambda_consistency: Weight for output consistency loss
        learning_rate: Learning rate for this stage
    """
    name: str
    epochs: int
    lambda_teacher: float
    lambda_dominance: float
    lambda_residual: float
    lambda_consistency: float
    learning_rate: float


class TrainingCurriculum:
    """
    Progressive training curriculum for adapter training.

    Three-stage curriculum:
    - Stage 1 (alignment): Learn basic alignment with semantic merge
    - Stage 2 (dominance): Add feature dominance constraints
    - Stage 3 (full): Add output consistency and decay teacher loss
    """

    def __init__(self, custom_stages: List[CurriculumStage] = None):
        """
        Initialize training curriculum.

        Args:
            custom_stages: Optional custom curriculum stages.
                          If None, uses default 3-stage curriculum.
        """
        if custom_stages is not None:
            self.stages = custom_stages
        else:
            self.stages = self._create_default_stages()

        self.current_stage_idx = 0
        self.current_epoch = 0

    def _create_default_stages(self) -> List[CurriculumStage]:
        """
        Create default 3-stage curriculum.

        Returns:
            List of curriculum stages
        """
        return [
            # Stage 1: Basic alignment (epochs 0-50)
            # Focus on matching teacher output
            CurriculumStage(
                name="alignment",
                epochs=50,
                lambda_teacher=1.0,
                lambda_dominance=0.0,
                lambda_residual=0.1,
                lambda_consistency=0.0,
                learning_rate=1e-4,
            ),
            # Stage 2: Add feature dominance (epochs 50-100)
            # Learn to respect feature assignments
            CurriculumStage(
                name="dominance",
                epochs=50,
                lambda_teacher=1.0,
                lambda_dominance=0.5,
                lambda_residual=0.1,
                lambda_consistency=0.0,
                learning_rate=5e-5,
            ),
            # Stage 3: Full training (epochs 100-150)
            # Add output consistency and reduce teacher reliance
            CurriculumStage(
                name="full",
                epochs=50,
                lambda_teacher=0.5,  # Decay teacher
                lambda_dominance=1.0,
                lambda_residual=0.1,
                lambda_consistency=0.3,
                learning_rate=1e-5,
            ),
        ]

    def get_current_stage(self) -> CurriculumStage:
        """
        Get current curriculum stage.

        Returns:
            Current CurriculumStage
        """
        return self.stages[self.current_stage_idx]

    def step_epoch(self) -> bool:
        """
        Step to next epoch.

        Returns:
            True if advanced to next stage, False otherwise
        """
        self.current_epoch += 1

        # Calculate total epochs up to and including current stage
        epochs_in_current_stage = sum(
            stage.epochs for stage in self.stages[:self.current_stage_idx + 1]
        )

        # Check if we should advance to next stage
        if self.current_epoch >= epochs_in_current_stage:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                return True

        return False

    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get current loss weights from active stage.

        Returns:
            Dict with lambda_teacher, lambda_dominance, lambda_residual, lambda_consistency
        """
        stage = self.get_current_stage()
        return {
            "lambda_teacher": stage.lambda_teacher,
            "lambda_dominance": stage.lambda_dominance,
            "lambda_residual": stage.lambda_residual,
            "lambda_consistency": stage.lambda_consistency,
        }

    def get_learning_rate(self) -> float:
        """
        Get current learning rate from active stage.

        Returns:
            Learning rate (float)
        """
        return self.get_current_stage().learning_rate

    def get_total_epochs(self) -> int:
        """
        Get total number of epochs across all stages.

        Returns:
            Total epochs (int)
        """
        return sum(stage.epochs for stage in self.stages)

    def get_progress(self) -> Dict[str, any]:
        """
        Get training progress information.

        Returns:
            Dict with stage_name, stage_index, current_epoch, total_epochs, stage_progress
        """
        stage = self.get_current_stage()

        # Calculate epochs at start of current stage
        epochs_before_current_stage = sum(
            s.epochs for s in self.stages[:self.current_stage_idx]
        )

        # Calculate progress within current stage
        epoch_in_stage = self.current_epoch - epochs_before_current_stage
        stage_progress = epoch_in_stage / stage.epochs if stage.epochs > 0 else 1.0

        return {
            "stage_name": stage.name,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "current_epoch": self.current_epoch,
            "total_epochs": self.get_total_epochs(),
            "epoch_in_stage": epoch_in_stage,
            "stage_epochs": stage.epochs,
            "stage_progress": stage_progress,
        }

    def reset(self):
        """Reset curriculum to beginning."""
        self.current_stage_idx = 0
        self.current_epoch = 0

    def __repr__(self) -> str:
        progress = self.get_progress()
        return (
            f"TrainingCurriculum("
            f"stage={progress['stage_name']}, "
            f"epoch={progress['current_epoch']}/{progress['total_epochs']})"
        )
