"""Real-Time Action Chunking (RTC) inference algorithm.

This module implements the RTC inference algorithm for online RL,
using Ray actors for distributed execution with SharedMemory for
efficient state management.

Components:
- SharedMemoryManager: Direct SharedMemory access for state management
- InferenceActor: GPU-resident inference with guided action chunking
- ControllerActor: Robot I/O and episode recording

The SharedMemoryManager replaces the Ray-based RTCStateActor to eliminate
Ray actor communication overhead for high-frequency state operations.
"""
from .data_manager.igris_b.shm_manager_bridge import SharedMemoryManager
from .data_manager.utils.utils import ShmArraySpec, create_shared_ndarray
from .inference_actor import InferenceActor
from .control_actor import ControllerActor

__all__ = [
    "SharedMemoryManager",
    "ShmArraySpec",
    "create_shared_ndarray",
    "InferenceActor",
    "ControllerActor",
]
