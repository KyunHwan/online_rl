"""
Teleop provider abstraction and igris_b implementation.

TeleopProvider is the abstract interface for teleop action sources.
IgrisBTeleopProvider directly owns DxlMasterArm and ManusUDPReceiver nodes,
reading cached hardware data without ROS pub/sub overhead.
"""

from abc import ABC, abstractmethod
import numpy as np
from rclpy.executors import SingleThreadedExecutor

from env_actor.human_in_the_loop.teleoperation.igris_b.arms_dynamixel import DxlMasterArm
from env_actor.human_in_the_loop.teleoperation.igris_b.hands_manus import ManusUDPReceiver


class TeleopProvider(ABC):
    """Abstract interface for teleop action sources. Implementations must be non-blocking."""

    @abstractmethod
    def get_latest_action(self) -> np.ndarray | None:
        """Return latest 24D teleop action or None if unavailable. Must never block."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if teleop hardware is sending data."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up hardware resources."""
        ...


class IgrisBTeleopProvider(TeleopProvider):
    """
    Directly owns DxlMasterArm and ManusUDPReceiver nodes.
    Reads hardware data from cached values updated by their timer callbacks.

    DxlMasterArm reads 12 arm servo positions at 25Hz into self.q.
    ManusUDPReceiver receives glove UDP data and processes into 12D normalized fingers.

    get_latest_action() returns 24D: [left_joint*6, right_joint*6, left_finger*6, right_finger*6]
    matching SequentialActor's action format.
    """

    def __init__(self, executor: SingleThreadedExecutor, operator_name: str = 'default'):
        # Reuse DxlMasterArm — reads servos at 25Hz, caches in self.q
        self._dxl = DxlMasterArm(['right_arm', 'left_arm'])
        executor.add_node(self._dxl)

        # Reuse ManusUDPReceiver — receives UDP at ~100Hz, caches glove data
        self._manus = ManusUDPReceiver(operator_name=operator_name)
        executor.add_node(self._manus)

    def get_latest_action(self) -> np.ndarray | None:
        arm = self._dxl.get_joint_positions()       # [right*6, left*6] radians
        finger = self._manus.get_target_fingers()   # [right*6, left*6] normalized
        if arm is None or finger is None:
            return None
        # Reorder from [right, left] to [left, right] to match SequentialActor action format
        return np.concatenate([arm[6:], arm[:6], finger[6:], finger[:6]])

    def is_connected(self) -> bool:
        return (self._dxl.get_joint_positions() is not None and
                self._manus.get_target_fingers() is not None)

    def shutdown(self) -> None:
        self._dxl.shutdown()
        self._dxl.destroy_node()
        self._manus.destroy_node()
