"""
ActionMux — selects between policy and teleop action each tick.

Never blocks. Falls back to policy action if teleop data is unavailable.
"""

import numpy as np

from .teleop_provider import TeleopProvider
from .intervention_switch import InterventionSwitch, ControlMode


class ActionMux:
    """Selects between policy and teleop action each tick. Never blocks."""


    def __init__(self, teleop_provider: TeleopProvider, intervention_switch: InterventionSwitch):
        self._teleop = teleop_provider
        self._switch = intervention_switch

    def select(self, policy_action: np.ndarray) -> tuple[np.ndarray, ControlMode]:
        """
        Select action based on current control mode.

        Args:
            policy_action: 24D action from policy inference.

        Returns:
            (selected_action, control_mode) — the action to send to the robot
            and which mode produced it.
        """
        mode = self._switch.get_control_mode()

        if mode == ControlMode.TELEOP:
            teleop_action = self._teleop.get_latest_action()
            if teleop_action is not None:
                return teleop_action, ControlMode.TELEOP
            # Fallback to policy if teleop data unavailable
            return policy_action, ControlMode.POLICY

        return policy_action, ControlMode.POLICY
    
    def set_control_mode_to_policy(self):
        self._switch.set_control_mode_to_policy()
