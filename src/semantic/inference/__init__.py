"""
Inference pipeline for semantic adapters.

Provides inference-time adapter application without retraining.
"""

from .pipeline import AdapterInferencePipeline

__all__ = ["AdapterInferencePipeline"]
