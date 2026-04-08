"""Resfit residual policy for env_actor inference.

Wraps the trained Residual_Actor into the Policy protocol, providing the
`.inference(base_action, obs_data)` method that the RTC control loop expects.

The control loop calls:
    action = residual_policy.inference(base_policy_action, obs_data) + base_policy_action

This class converts numpy inputs to torch tensors, runs a single forward pass
through the Residual_Actor, and returns the residual delta as a numpy array
broadcast to match the action chunk shape.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from env_actor.policy.registry import POLICY_REGISTRY
from env_actor.inference_engine_utils.action_inpainting import compute_guided_prefix_weights

_CAMERAS = ("head", "left", "right")


@POLICY_REGISTRY.register("resfit_policy")
class ResidualPolicy:
    """Policy that applies a learned residual on top of a base action chunk.

    Args:
        components: Dict of GraphModel components built by the factory.
            Expected keys: ``resfit_residual_actor``.
        checkpoint_path: Optional directory containing ``resfit_residual_actor.pt``.
        obs_proprio_history: Number of proprioception timesteps expected in
            ``obs["proprio"]``. Must match the value used during training.
        **kwargs: Ignored (forward-compat with loader passing extra params).
    """

    def __init__(
        self,
        components: dict[str, nn.Module],
        *,
        checkpoint_path: str | None = None,
        obs_proprio_history: int = 1,
        **kwargs: Any,
    ) -> None:
        self.components = components
        self.obs_proprio_history = obs_proprio_history

        if checkpoint_path:
            pt = os.path.join(checkpoint_path, "resfit_residual_actor.pt")
            if os.path.exists(pt):
                state = torch.load(pt, map_location="cpu")
                components["resfit_residual_actor"].load_state_dict(state)

    # ------------------------------------------------------------------
    # Core inference (called by control loop)
    # ------------------------------------------------------------------

    def inference(
        self,
        base_action: np.ndarray,
        obs_data: dict[str, Any],
    ) -> np.ndarray:
        """Compute action residual from current observation and base action.

        Args:
            base_action: (action_chunk_size, action_dim) float32 numpy — base
                policy action chunk from the inference loop.
            obs_data: dict with keys:
                - ``proprio``: (obs_history, state_dim) float32 numpy
                - ``head``, ``left``, ``right``: (num_img_obs, 3, H, W) uint8

        Returns:
            (action_chunk_size, action_dim) float32 numpy — residual delta to
            be added to base_action by the control loop.
        """
        
        device = self._device()
        if base_action.ndim == 1:
            base_action = base_action.reshape(1, -1)
        action_chunk_size, action_dim = base_action.shape

        # 1. Prepare state dict with batch dim = 1
        state: dict[str, torch.Tensor] = {}

        # Camera images: take latest frame → (1, 3, H, W) float32
        for cam in _CAMERAS:
            if cam in obs_data:
                img = obs_data[cam] if obs_data[cam].ndim == 3 else obs_data[cam][-1]  # (3, H, W)
                state[cam] = torch.from_numpy(img).float().unsqueeze(0).to(device)

        # Proprio: single timestep → (1, 1, state_dim)
        state["proprio"] = (
            torch.from_numpy(obs_data["proprio"]).float().reshape(1, 1, -1).to(device)
        )

        # 2. Action: (1, 1, action_dim) — single action step for the actor
        #    Residual_Actor preprocessor expects (batch, 1, action_dim)
        action_tensor = (
            torch.from_numpy(base_action[0:1])
            .float()
            .unsqueeze(0)  # (1, 1, action_dim)
            .to(device)
        )

        # 3. Forward through Residual_Actor → (1, action_dim)
        delta = self.components["resfit_residual_actor"](
            state=state, action=action_tensor
        )

        return delta[0].cpu().float().numpy().copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        try:
            return next(
                self.components["resfit_residual_actor"].parameters()
            ).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Policy protocol
    # ------------------------------------------------------------------

    def predict(
        self, input_data: dict[str, Any], data_normalization_interface
    ) -> np.ndarray:
        base_action = input_data.get("base_action", np.zeros((5, 24), dtype=np.float32))
        return self.inference(base_action, input_data) + base_action

    def guided_inference(
        self,
        input_data: dict[str, Any],
        data_normalization_interface,
        min_num_actions_executed: int,
        action_chunk_size: int,
    ) -> np.ndarray:
        base_action = input_data.get("base_action", np.zeros((action_chunk_size, 24), dtype=np.float32))
        pred_actions = self.inference(base_action, input_data) + base_action

        weights = compute_guided_prefix_weights(
            input_data["est_delay"],
            min_num_actions_executed,
            action_chunk_size,
            schedule="exp",
        ).reshape(-1, 1)

        return input_data["prev_action"] * weights + pred_actions * (1.0 - weights)

    def warmup(self) -> None:
        device = self._device()
        proprio_state_dim = 24

        dummy_obs: dict[str, Any] = {
            "proprio": np.zeros(
                (self.obs_proprio_history, proprio_state_dim), dtype=np.float32
            ),
        }
        for cam in _CAMERAS:
            dummy_obs[cam] = np.zeros((1, 3, 240, 320), dtype=np.uint8)

        dummy_action = np.zeros((5, 24), dtype=np.float32)

        with torch.inference_mode():
            self.inference(dummy_action, dummy_obs)

    def freeze_all_model_params(self) -> None:
        for component in self.components.values():
            for p in component.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # nn.Module delegation (env_actor calls .eval() and .to(device))
    # ------------------------------------------------------------------

    def eval(self) -> "ResidualPolicy":
        for component in self.components.values():
            component.eval()
        return self

    def to(self, device: torch.device | str) -> "ResidualPolicy":
        for component in self.components.values():
            component.to(device)
        return self

    def parameters(self):
        for component in self.components.values():
            yield from component.parameters()

    def state_dict(self) -> dict:
        return {k: v.state_dict() for k, v in self.components.items()}

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        for k, v in self.components.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k], strict=strict)
