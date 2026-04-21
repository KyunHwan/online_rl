"""igris_b action interpolator.

24D action layout:
  [0:12]  = arm joint angles (radians) — shortest-arc interp
  [12:24] = finger targets (normalized [0,1]) — linear interp
"""

import numpy as np

from env_actor.human_in_the_loop.action_mux.interp_utils.interp_interface import (
    ActionInterpolator,
)


def _angle_diff(target, source):
    """Signed shortest angular difference, wrapped to [-pi, pi]."""
    return (target - source + np.pi) % (2 * np.pi) - np.pi


class IgrisBInterpolator(ActionInterpolator):
    JOINT_DIMS = slice(0, 12)
    FINGER_DIMS = slice(12, 24)
    ACTION_DIM = 24

    def interpolate(self, action_from, action_to, num_steps=5, epsilon=1e-3):
        a_from = np.asarray(action_from, dtype=np.float64)
        a_to = np.asarray(action_to, dtype=np.float64)

        joint_delta = _angle_diff(a_to[self.JOINT_DIMS], a_from[self.JOINT_DIMS])
        finger_delta = a_to[self.FINGER_DIMS] - a_from[self.FINGER_DIMS]

        trajectory = []
        for t in range(1, num_steps + 1):
            alpha = t / num_steps
            step = np.empty(self.ACTION_DIM, dtype=np.float64)

            j = a_from[self.JOINT_DIMS] + alpha * joint_delta
            step[self.JOINT_DIMS] = (j + np.pi) % (2 * np.pi) - np.pi

            f = a_from[self.FINGER_DIMS] + alpha * finger_delta
            step[self.FINGER_DIMS] = np.clip(f, 0.0, 1.0)

            trajectory.append(step)

            joint_err = np.max(np.abs(_angle_diff(a_to[self.JOINT_DIMS], step[self.JOINT_DIMS])))
            finger_err = np.max(np.abs(a_to[self.FINGER_DIMS] - step[self.FINGER_DIMS]))
            if joint_err < epsilon and finger_err < epsilon:
                break
        return trajectory
