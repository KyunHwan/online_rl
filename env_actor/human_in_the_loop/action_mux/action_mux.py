"""
ActionMux — selects between policy and teleop action each tick.

Smoothly interpolates on every POLICY<->TELEOP transition using a
robot-specific ActionInterpolator owned by the mux. Never blocks.
Falls back to policy action if teleop data is unavailable.
"""

import numpy as np

from .teleop_provider import TeleopProvider
from .intervention_switch import InterventionSwitch, ControlMode
from .interp_utils.interp_interface import ActionInterpolator


def _build_interpolator(robot: str) -> ActionInterpolator:
    if robot == "igris_b":
        from .interp_utils.robots.igris_b import IgrisBInterpolator
        return IgrisBInterpolator()
    raise ValueError(f"Unknown robot for ActionMux interpolator: {robot}")


class ActionMux:
    """Selects between policy and teleop each tick with a smooth blend on transitions. Never blocks."""

    def __init__(
        self,
        teleop_provider: TeleopProvider,
        intervention_switch: InterventionSwitch,
        robot: str,
        interp_steps: int = 20,
        interp_epsilon: float = 1e-3,
    ):
        self._teleop = teleop_provider
        self._switch = intervention_switch
        self._interp = _build_interpolator(robot)
        self._interp_steps = interp_steps
        self._interp_epsilon = interp_epsilon

        self._prev_mode: ControlMode | None = None
        self._last_emitted: np.ndarray | None = None
        self._traj: list[np.ndarray] = []
        self._traj_dest_mode: ControlMode | None = None

    def select(self, policy_action: np.ndarray) -> tuple[np.ndarray, ControlMode]:
        """
        Select action based on current control mode, with an interpolated
        handoff the first tick after each mode transition.

        Args:
            policy_action: 24D action from policy inference.

        Returns:
            (selected_action, control_mode) — the action to send to the robot
            and which mode produced it (destination mode during a blend).
        """
        current_mode = self._switch.get_control_mode()

        if current_mode == ControlMode.TELEOP:
            teleop_action = self._teleop.get_latest_action()
            if teleop_action is not None:
                target_action = teleop_action
                effective_mode = ControlMode.TELEOP
            else:
                print("Teleop Action is None !!!")
                target_action = policy_action
                effective_mode = ControlMode.POLICY
        else:
            target_action = policy_action
            effective_mode = ControlMode.POLICY

        if self._last_emitted is None or self._prev_mode is None:
            self._traj = []
            self._traj_dest_mode = None
            self._last_emitted = np.asarray(target_action, dtype=np.float64).copy()
            self._prev_mode = effective_mode
            return target_action, effective_mode

        if effective_mode != self._prev_mode:
            if self._prev_mode == ControlMode.POLICY and effective_mode == ControlMode.TELEOP:
                self._traj = self._interp.interpolate(
                    self._last_emitted, target_action,
                    num_steps=self._interp_steps, epsilon=self._interp_epsilon,
                )
                self._traj_dest_mode = effective_mode
            else:
                # TELEOP -> POLICY (including teleop-None fallback): no blend, hand back raw target.
                self._traj = []
                self._traj_dest_mode = None
        elif not (effective_mode == self._traj_dest_mode and self._traj):
            self._traj = []
            self._traj_dest_mode = None

        if self._traj:
            step = self._traj.pop(0)
            self._last_emitted = step
            self._prev_mode = effective_mode
            return step, self._traj_dest_mode

        self._last_emitted = np.asarray(target_action, dtype=np.float64).copy()
        self._prev_mode = effective_mode
        return target_action, effective_mode

    def set_control_mode_to_policy(self):
        self._switch.set_control_mode_to_policy()
        self._traj = []
        self._traj_dest_mode = None
        self._last_emitted = None
        self._prev_mode = None
