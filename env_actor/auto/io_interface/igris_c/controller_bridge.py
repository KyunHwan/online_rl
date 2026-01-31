"""
IGRIS_C Controller Bridge - Interface Stub

This is a skeleton implementation documenting the required interface.
Complete implementation pending IGRIS_C hardware specifications and communication protocol.

Reference: /env_actor/auto/io_interface/igris_b/controller_bridge.py
"""

import numpy as np


class ControllerBridge:
    """
    Stateless controller bridging between IGRIS_C robot and the controller interface.

    REQUIRED INTERFACE METHODS (must match IGRIS_B interface):
    - __init__(inference_runtime_config)
    - read_state() -> dict
    - publish_action(action, prev_joint) -> np.ndarray
    - start_state_readers() -> None
    - init_robot_position() -> np.ndarray
    - recorder_rate_controller() -> rate object

    REQUIRED PROPERTIES:
    - DT: float (timestep in seconds, = 1.0 / HZ)
    - policy_update_period: int
    """

    def __init__(self, inference_runtime_config):
        """
        Initialize controller bridge for IGRIS_C.

        TODO: Implement based on IGRIS_C communication protocol
        Args:
            inference_runtime_config: Runtime configuration dict
        """
        raise NotImplementedError("IGRIS_C controller bridge implementation pending hardware specs")

    @property
    def DT(self):
        """Return control timestep (1.0 / HZ)."""
        raise NotImplementedError()

    @property
    def policy_update_period(self):
        """Return policy update period from runtime params."""
        raise NotImplementedError()

    def recorder_rate_controller(self):
        """Return rate controller for maintaining control frequency."""
        raise NotImplementedError()

    def read_state(self) -> dict:
        """
        Read current robot state (proprioception + images).

        Returns:
            Dict with 'proprio' key (np.ndarray) and camera keys
        """
        raise NotImplementedError()

    def publish_action(self, action: np.ndarray, prev_joint: np.ndarray) -> np.ndarray:
        """
        Publish action to robot with slew-rate limiting.

        Returns:
            Smoothed joint positions
        """
        raise NotImplementedError()

    def start_state_readers(self):
        """Start background readers for state acquisition."""
        raise NotImplementedError()

    def init_robot_position(self) -> np.ndarray:
        """Initialize robot to safe starting position."""
        raise NotImplementedError()
