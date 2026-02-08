"""
Intervention switch abstraction and pedal implementation.

InterventionSwitch determines the current control mode (POLICY vs TELEOP).
PedalInterventionSwitch listens to keyboard events via the io_event ROS topic.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class ControlMode(IntEnum):
    POLICY = 0
    TELEOP = 1


class InterventionSwitch(ABC):
    """Abstract interface for determining control mode. Must be non-blocking."""

    @abstractmethod
    def get_control_mode(self) -> ControlMode:
        """Return current control mode. Must never block."""
        ...


class PedalInterventionSwitch(InterventionSwitch):
    """
    Subscribes to /igris_b/{robot_id}/io_event and toggles between POLICY and TELEOP.

    Key events (from pedal_publisher.py IOHandler):
      '$' -> toggle POLICY <-> TELEOP (instant, no sleep)
      '^' -> force back to POLICY
    """

    def __init__(self, ros_node: Node, robot_id: str):
        self._mode = ControlMode.POLICY
        self._lock = threading.Lock()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        ros_node.create_subscription(
            String,
            f'/igris_b/{robot_id}/io_event',
            self._io_cb,
            qos,
        )

    def _io_cb(self, msg: String) -> None:
        event = msg.data.strip().strip("'\"")
        with self._lock:
            if event == "$":
                self._mode = (ControlMode.POLICY
                              if self._mode == ControlMode.TELEOP
                              else ControlMode.TELEOP)
            elif event == "^":
                self._mode = ControlMode.POLICY

    def get_control_mode(self) -> ControlMode:
        with self._lock:
            return self._mode
        
    def set_control_mode_to_policy(self):
        with self._lock:
            self._mode = ControlMode.POLICY
