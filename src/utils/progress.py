"""
Thread-safe progress tracking for LoRA Power-Merger.

Provides ThreadSafeProgressBar wrapper for ComfyUI progress bars.
"""

import threading
from typing import Optional
import comfy.utils


class ThreadSafeProgressBar:
    """
    Thread-safe wrapper for ComfyUI progress bars.

    Ensures progress updates from multiple threads don't cause race conditions.
    """

    def __init__(self, total: int, desc: str = "Processing"):
        """
        Initialize thread-safe progress bar.

        Args:
            total: Total number of steps
            desc: Description for progress bar
        """
        self.total = total
        self.desc = desc
        self._lock = threading.Lock()
        self._current = 0
        self._pbar = comfy.utils.ProgressBar(total)

    def update(self, n: int = 1):
        """
        Update progress by n steps (thread-safe).

        Args:
            n: Number of steps to increment
        """
        with self._lock:
            self._current += n
            self._pbar.update(n)

    def set_description(self, desc: str):
        """
        Update progress bar description (thread-safe).

        Args:
            desc: New description
        """
        with self._lock:
            self.desc = desc
            # ComfyUI progress bars don't support dynamic descriptions
            # but we store it for potential logging

    def reset(self):
        """Reset progress to zero (thread-safe)."""
        with self._lock:
            self._current = 0
            self._pbar = comfy.utils.ProgressBar(self.total)

    def close(self):
        """Close the progress bar (thread-safe)."""
        with self._lock:
            # Ensure we're at 100%
            if self._current < self.total:
                remaining = self.total - self._current
                self._pbar.update(remaining)

    @property
    def current(self) -> int:
        """Get current progress value (thread-safe)."""
        with self._lock:
            return self._current

    @property
    def percentage(self) -> float:
        """Get current progress percentage (thread-safe)."""
        with self._lock:
            if self.total == 0:
                return 0.0
            return (self._current / self.total) * 100.0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
