"""Abstract interface for action interpolation between control modes.

Concrete robot implementations encapsulate action-layout specifics
(which dims are angles, which are normalized, etc.).
"""

from abc import ABC, abstractmethod

import numpy as np


class ActionInterpolator(ABC):
    """Computes a smooth trajectory from one action vector to another.

    Robot-specific implementations know their action layout and apply the
    right interpolation kind per dim group (e.g., shortest-arc for joint
    angles, linear for normalized finger positions).
    """

    @abstractmethod
    def interpolate(
        self,
        action_from: np.ndarray,
        action_to: np.ndarray,
        num_steps: int = 5,
        epsilon: float = 1e-3,
    ) -> list[np.ndarray]:
        """Return intermediate actions from `action_from` to `action_to`.

        - Excludes `action_from`; includes the endpoint (or an early-stop
          pose within `epsilon` of `action_to`).
        - Length <= `num_steps`.
        - Must be pure and non-blocking.
        """
        ...
