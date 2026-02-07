"""
Sequential inference algorithm for robot manipulation.

This module provides robot-agnostic sequential inference with modular components
that can be reused for other inference algorithms (e.g., RTC).
"""

from .sequential_actor import SequentialActor

__all__ = ["SequentialActor"]
