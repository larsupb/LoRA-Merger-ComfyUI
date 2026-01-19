"""
Serialization utilities for adapter packages.

Provides save/load functionality for trained adapters in SafeTensors format.
"""

from .format import AdapterSerializer

__all__ = ["AdapterSerializer"]
