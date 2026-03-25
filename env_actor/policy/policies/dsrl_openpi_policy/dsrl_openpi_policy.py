"""DSRL + OpenPI policy for env_actor inference.

Wires four trained components into the Policy protocol:
  1. backbone (RadioV3)                       — raw images → spatial features
  2. noise_processor (NoiseActorImgDepthProprioProcessor) — features + proprio → flat latent
  3. noise_actor (Noise_Latent_Actor)         — flat latent → structured noise
  4. openpi_model (OpenPiBatchedWrapper)      — images + proprio + noise → action chunk

All inference goes through each component's GraphModel.forward(); no module unwrapping.
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


@POLICY_REGISTRY.register("dsrl_openpi_policy")
class DsrlOpenpiPolicy:
    """Policy that combines a DSRL noise-latent actor with OpenPI diffusion.

    Args:
        components: Dict of GraphModel components built by the factory.
            Expected keys: ``backbone``, ``noise_processor``, ``noise_actor``,
            ``openpi_model``.
        checkpoint_path: Optional directory containing ``backbone.pt``,
            ``noise_processor.pt``, and ``noise_actor.pt`` for the DSRL
            components. ``openpi_model`` loads its own weights via ``ckpt_dir``
            in its component YAML, so it is intentionally excluded here.
        obs_proprio_history: Number of proprioception timesteps expected in
            ``obs["proprio"]``. Must match the value used during training (50).
        **kwargs: Ignored (forward-compat with loader passing extra params).
    """

    def __init__(
        self,
        components: dict[str, nn.Module],
        *,
        checkpoint_path: str | None = None,
        obs_proprio_history: int = 50,
        **kwargs: Any,
    ) -> None:
        self.components = components
        self.obs_proprio_history = obs_proprio_history

        # Load DSRL component weights. OpenPI manages its own weights via
        # ckpt_dir specified inside its component YAML.
        if checkpoint_path:
            for name in ("backbone", "noise_processor", "noise_actor"):
                pt = os.path.join(checkpoint_path, f"{name}.pt")
                if os.path.exists(pt):
                    state = torch.load(pt, map_location="cpu")
                    components[name].load_state_dict(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(
        self, obs: dict[str, Any], data_normalization_interface
    ) -> np.ndarray:
        """Core forward pass. Returns (action_horizon, action_dim) float32."""
        device = self._device()

        # 1. Normalize full proprio history and move to device.
        #    proprio shape from shm: (obs_proprio_history, state_dim)
        #    DataNormalizationBridge.normalize_state() expects a dict with a
        #    'proprio' key; returns the same dict with the field normalized.
        proprio_raw = obs["proprio"]  # (H, D) float32 numpy
        proprio_norm = data_normalization_interface.normalize_state(
            {"proprio": proprio_raw}
        )["proprio"]
        proprio_batch = (
            torch.from_numpy(proprio_norm).float().unsqueeze(0).to(device)
        )  # (1, H, D)

        # 2. Prepare latest raw images as float32 tensors on device (for backbone).
        img_tensors: dict[str, torch.Tensor] = {}
        for cam in _CAMERAS:
            if cam in obs:
                # obs[cam]: (num_img_obs, 3, H, W) uint8 or float32
                img = obs[cam][-1]  # (3, H, W)
                img_tensors[cam] = (
                    torch.from_numpy(img).float().unsqueeze(0).to(device)
                )  # (1, 3, H, W)

        # 3. Backbone: raw images → spatial feature maps.
        #    backbone GraphModel input name: "image"
        #    backbone returns (features, summary); we need features → (1, 1024, H', W')
        feat: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for cam in _CAMERAS:
                if cam in img_tensors:
                    features, _ = self.components["backbone"](image=img_tensors[cam])
                    feat[cam] = features  # (1, 1024, H', W')

        # 4. Noise processor: features + normalized proprio → flat latent.
        #    noise_processor GraphModel input name: "data"
        proc_input = {**feat, "proprio": proprio_batch}
        flat = self.components["noise_processor"](data=proc_input)  # (1, D)

        # 5. Noise actor: flat latent → structured noise.
        #    noise_actor GraphModel input name: "flat_features"
        noise = self.components["noise_actor"](flat_features=flat)  # (1, 50, 32)

        # 6. OpenPI: raw images + raw proprio[0] + noise → action chunk.
        #    Matches action_critic_trainer.py lines 70-77 exactly:
        #      - proprio is raw (un-normalized), single latest step
        #      - images are float32 numpy arrays
        #      - noise is detached cpu float32 numpy
        openpi_obs: dict[str, Any] = {
            "proprio": obs["proprio"][0:1].astype(np.float32),  # (1, D) raw
        }
        for cam in _CAMERAS:
            if cam in obs:
                openpi_obs[cam] = obs[cam][-1:].astype(np.float32)  # (1, 3, H, W)
        if "prompt" in obs:
            openpi_obs["prompt"] = obs["prompt"]

        noise_np = noise.detach().cpu().float().numpy()  # (1, 50, 32)
        # Generates Gaussian noise
        noise_np = np.random.randn(1, 50, 32)
        
        # openpi_model GraphModel inputs: [observation, noise]
        actions = self.components["openpi_model"](
            observation=openpi_obs, noise=noise_np
        )  # (1, action_horizon, action_dim) tensor

        return actions[0].cpu().float().numpy()  # (action_horizon, action_dim)

    def _device(self) -> torch.device:
        """Infer device from backbone parameters."""
        try:
            return next(self.components["backbone"].parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Policy protocol
    # ------------------------------------------------------------------

    def predict(
        self, obs: dict[str, Any], data_normalization_interface
    ) -> np.ndarray:
        """Single-sample inference.

        Returns:
            np.float32 array of shape (action_horizon, action_dim).
        """
        return self._run_inference(obs, data_normalization_interface)

    def guided_inference(
        self,
        input_data: dict[str, Any],
        data_normalization_interface,
        min_num_actions_executed: int,
        action_chunk_size: int,
    ) -> np.ndarray:
        """RTC inference with action-inpainting blending.

        Blends freshly predicted actions with the unexecuted tail of the
        previous action chunk using an exponential weight schedule.

        Returns:
            np.float32 array of shape (action_chunk_size, action_dim).
        """
        pred_actions = self._run_inference(input_data, data_normalization_interface)

        weights = compute_guided_prefix_weights(
            input_data["est_delay"],
            min_num_actions_executed,
            action_chunk_size,
            schedule="exp",
        ).reshape(-1, 1)

        return input_data["prev_action"] * weights + pred_actions * (1.0 - weights)

    def warmup(self) -> None:
        """Run one dummy forward pass to trigger CUDA graph/compile warmup."""
        device = self._device()
        proprio_state_dim = 24  # standard for igris_b

        dummy_obs: dict[str, Any] = {
            "proprio": np.zeros(
                (self.obs_proprio_history, proprio_state_dim), dtype=np.float32
            ),
            "prev_action": np.zeros((50, proprio_state_dim), dtype=np.float32),
            "est_delay": 0,
        }
        # Use 320x240 to match default mono_image_resize in runtime config
        for cam in _CAMERAS:
            dummy_obs[cam] = np.zeros((1, 3, 240, 320), dtype=np.uint8)

        class _IdentityNorm:
            def normalize_state(self, x):
                return x.copy()

        with torch.inference_mode():
            self._run_inference(dummy_obs, _IdentityNorm())

    def freeze_all_model_params(self) -> None:
        """Freeze every parameter across all components."""
        for component in self.components.values():
            for p in component.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # nn.Module delegation (env_actor calls .eval() and .to(device))
    # ------------------------------------------------------------------

    def eval(self) -> "DsrlOpenpiPolicy":
        for component in self.components.values():
            component.eval()
        return self

    def to(self, device: torch.device | str) -> "DsrlOpenpiPolicy":
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
