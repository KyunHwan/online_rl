"""
IGRIS_C Data Manager Bridge - Interface Stub

This is a skeleton implementation documenting the required interface.
Complete implementation pending IGRIS_C hardware specifications.

Reference: /env_actor/auto/data_manager/igris_b/data_manager_bridge.py
"""

import numpy as np
import torch


class DataManagerBridge:
    """
    Stateful data manager for IGRIS_C.

    REQUIRED INTERFACE METHODS (must match IGRIS_B interface):
    - __init__(inference_runtime_config)
    - update_norm_stats()
    - update_prev_joint(val)
    - update_state_history(obs_data)
    - denormalize_action(action, device) -> np.ndarray
    - serve_normalized_obs_state(device) -> dict
    - generate_noise(device) -> torch.Tensor
    - buffer_action_chunk(policy_output, t, device)
    - get_current_action(t) -> np.ndarray
    - init_inference_obs_state_buffer(init_data)
    - init_train_data_buffer()

    REQUIRED PROPERTIES:
    - state_dim: int (dimension of proprioceptive state)
    - prev_joint: np.ndarray (previous joint positions)
    """

    def __init__(self, inference_runtime_config):
        """
        Initialize data manager bridge for IGRIS_C.

        TODO: Implement based on IGRIS_C specifications
        Args:
            inference_runtime_config: Runtime configuration dict
        """
        raise NotImplementedError("IGRIS_C data manager bridge implementation pending hardware specs")

    @property
    def prev_joint(self):
        """Return previous joint positions."""
        raise NotImplementedError()

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def update_prev_joint(self, val):
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def update_norm_stats(self):
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def update_state_history(self, obs_data):
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def serve_normalized_obs_state(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def generate_noise(self, device: torch.device) -> torch.Tensor:
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def buffer_action_chunk(self, policy_output: torch.Tensor, current_step: int):
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def get_current_action(self, current_step: int) -> np.ndarray:
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
    
    def init_inference_obs_state_buffer(self, init_data):
        """Update normalization stats (placeholder for dynamic stats updates)."""
        raise NotImplementedError()
