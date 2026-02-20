"""Policy interface for env_actor."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Policy(Protocol):
    """Interface for inference-time policies built from component modules."""

    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None: ...

    def predict(self, input_data: dict, data_normalization_interface):
        ...

    def guided_inference(self, input_data: dict, data_normalization_interface, min_num_actions_executed, action_chunk_size):
        """
        Uses inpainting technique from Real-Time Execution of Action Chunking Flow Policies
        https://arxiv.org/pdf/2506.07339
        """
        ...
        
    def warmup(self) -> None:
        """
        This is the warmup that is needed to choose the fastest CUDA algorithm.
        Used with torch.backends.cudnn.benchmark = True 
        """
        ...
    
    def freeze_all_model_params(self) -> None: 
        """
        Freezes every model parameter
        """
        ...

