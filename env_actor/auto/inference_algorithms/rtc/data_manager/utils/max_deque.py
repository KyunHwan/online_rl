"""MaxDeque utility for tracking maximum delay over a sliding window.

This is used by the RTC inference algorithm to estimate system latency
for guided action chunk refinement.
"""
from collections import deque


class MaxDeque:
    """Tracks maximum value over a sliding window of recent delays.

    Used by the inference loop to estimate system latency (number of control
    steps between inference calls) for the guided action chunk refinement.

    Args:
        buffer_len: Size of the sliding window (default: 5)
    """

    def __init__(self, buffer_len: int = 5):
        self.dq: deque = deque([])
        self.buffer_len = buffer_len

    def add(self, delay: int) -> None:
        """Add a new delay value to the buffer.

        If buffer is full, removes the oldest value first.

        Args:
            delay: Number of control iterations since last inference
        """
        if len(self.dq) == self.buffer_len:
            self.dq.popleft()
        self.dq.append(delay)

    def max(self) -> int:
        """Return the maximum delay in the buffer.

        Returns:
            Maximum delay value, or 0 if buffer is empty
        """
        return max(self.dq) if self.dq else 0

    def clear(self) -> None:
        """Clear all values from the buffer."""
        self.dq.clear()
