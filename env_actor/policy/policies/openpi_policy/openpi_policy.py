"""OpenPI policy wrapper for build_policy() integration.

Wraps OpenPiBatchedWrapper (built as a model component via the factory)
and adapts it to the Policy protocol used by env_actor inference loops.

Handles single-sample to batched conversion so that existing openpi actors
can call predict() with unbatched observations (matching Pi05IgrisVlaAdapter).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from env_actor.policy.registry import POLICY_REGISTRY


@POLICY_REGISTRY.register("openpi_policy")
class OpenPiPolicy:
    """Policy wrapper for OpenPiBatchedWrapper via build_policy().

    Receives the OpenPiBatchedWrapper as a model component built by the
    factory, and provides the Policy protocol interface expected by
    inference actors.

    Args:
        components: Dict of named nn.Module components. Expects a key
            whose GraphModel contains an ``openpi_model`` sub-module
            (the OpenPiBatchedWrapper).
        **kwargs: Additional policy parameters (currently unused).
    """

    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None:
        self._wrapper = self._resolve_wrapper(components)

        # Expose metadata from wrapper
        self.action_dim = self._wrapper.action_dim
        self.action_horizon = self._wrapper.action_horizon
        self.state_dim = self._wrapper.state_dim
        self.norm_stats = self._wrapper.norm_stats

    @staticmethod
    def _resolve_wrapper(components: dict[str, nn.Module]) -> nn.Module:
        """Extract the OpenPiBatchedWrapper from the components dict.

        The factory wraps each component YAML in a GraphModel. The actual
        OpenPiBatchedWrapper lives inside ``graph_model.graph_modules``.
        """
        # Get the first (and only) component
        if len(components) == 1:
            graph_model = next(iter(components.values()))
        elif "openpi_model" in components:
            graph_model = components["openpi_model"]
        else:
            raise ValueError(
                f"Cannot resolve OpenPiBatchedWrapper from components: "
                f"{list(components.keys())}. Expected single component or "
                f"'openpi_model' key."
            )

        # Unwrap from GraphModel
        if hasattr(graph_model, "graph_modules"):
            if "openpi_model" in graph_model.graph_modules:
                return graph_model.graph_modules["openpi_model"]
            # Fall back to single module in graph
            if len(graph_model.graph_modules) == 1:
                return next(iter(graph_model.graph_modules.values()))
            raise ValueError(
                f"Cannot resolve OpenPiBatchedWrapper from graph_modules: "
                f"{list(graph_model.graph_modules.keys())}. Expected "
                f"'openpi_model' key."
            )

        # If not a GraphModel, assume it's the wrapper directly
        return graph_model

    def predict(
        self, obs: dict[str, Any], noise: np.ndarray | None = None
    ) -> np.ndarray:
        """Run inference on single-sample observations.

        Accepts the same observation format as Pi05IgrisVlaAdapter.predict():
        - obs["proprio"]: (num_robot_obs, state_dim) float32
        - obs["head"/"left"/"right"]: (num_img_obs, 3, H, W) uint8

        Extracts the latest observation, adds a batch dimension, delegates
        to OpenPiBatchedWrapper.predict(), then removes the batch dimension.

        Returns:
            np.float32 array of shape (action_horizon, action_dim).
        """
        # Extract latest timestep and add batch dim
        batched_obs = {}
        batched_obs["proprio"] = obs["proprio"][-1:].reshape(1, -1)
        for cam in ("head", "left", "right"):
            if cam in obs:
                batched_obs[cam] = obs[cam][-1:].reshape(
                    1, *obs[cam][-1].shape
                )
        if "prompt" in obs:
            batched_obs["prompt"] = obs["prompt"]

        # Delegate to wrapper (returns (1, action_horizon, action_dim))
        batched_actions = self._wrapper.predict(batched_obs, noise=noise)

        # Remove batch dimension
        return batched_actions[0]

    def guided_inference(self, input_data: dict[str, Any]):
        """Not implemented — openpi actors handle blending externally."""
        raise NotImplementedError(
            "OpenPiPolicy does not support guided_inference. "
            "The openpi actors handle action chunk blending externally."
        )

    def warmup(self) -> None:
        """Trigger torch.compile warmup via the wrapper."""
        self._wrapper.warmup(batch_size=1)

    def freeze_all_model_params(self) -> None:
        """Freeze all model parameters."""
        self._wrapper.freeze_all_model_params()

    # -- Utility methods (delegate to wrapper) --

    def eval(self) -> "OpenPiPolicy":
        self._wrapper.eval()
        return self

    def to(self, device: torch.device) -> "OpenPiPolicy":
        self._wrapper.to(device)
        return self

    def parameters(self):
        return self._wrapper.parameters()

    def state_dict(self):
        return self._wrapper.state_dict()

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        return self._wrapper.load_state_dict(state_dict, strict=strict)
