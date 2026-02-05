"""VFP single expert policy for RTC and sequential inference.

This policy wraps component modules and provides the interface needed
for both sequential inference and real-time action chunking (RTC).

RTC-required methods:
- encode_memory(): Encode robot history and camera images
- body: Property returning the action decoder module
- freeze_all_model_params(): Freeze parameters for VJP
- normalization_tensors: Property returning normalization stats
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from env_actor.policy.registry import POLICY_REGISTRY


@POLICY_REGISTRY.register("vfp_single_expert")
class VFPSingleExpertPolicy:
    """Policy wrapper that holds component modules.

    Supports both sequential inference and RTC inference modes.

    For RTC, the policy must provide:
    - encode_memory(rh, cam_images): Encode observations into memory
    - body property: Return the action decoder for guided inference
    - freeze_all_model_params(): Freeze all parameters for VJP
    - normalization_tensors property: Return normalization stats

    Args:
        components: Dict of named nn.Module components
        **kwargs: Policy parameters including:
            - state_dim: Proprioception state dimensionality
            - action_dim: Action dimensionality
            - num_queries: Action chunk length
            - num_robot_observations: Robot observation history depth
            - num_image_observations: Image observation history depth
            - image_observation_skip: Frame skip for image updates
            - camera_names: List of camera identifiers
            - stats_path: Path to normalization stats file
            - stats_eps: Epsilon for normalization stability
    """

    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None:
        self.components = components
        self.config = kwargs

        # Resolve the main policy module
        self._policy = self._resolve_policy_component(components)

        # Extract dimensions from kwargs (passed from policy YAML)
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.num_queries = kwargs.get("num_queries")
        self.num_robot_observations = kwargs.get("num_robot_observations")
        self.num_image_observations = kwargs.get("num_image_observations")
        self.image_observation_skip = kwargs.get("image_observation_skip", 1)
        self.camera_names = kwargs.get("camera_names", [])
        self.stats_eps = kwargs.get("stats_eps", 1e-8)

        # Normalization tensors (loaded lazily)
        self._state_mean: Optional[torch.Tensor] = None
        self._state_std: Optional[torch.Tensor] = None
        self._action_mean: Optional[torch.Tensor] = None
        self._action_std: Optional[torch.Tensor] = None

        # Load normalization stats if path provided
        stats_path = kwargs.get("stats_path")
        if stats_path:
            self._load_normalization_stats(stats_path)

    def _resolve_policy_component(self, components: dict[str, nn.Module]) -> nn.Module:
        """Resolve the main policy module from components dict.

        Args:
            components: Dict of named modules

        Returns:
            The main policy module

        Raises:
            ValueError: If cannot resolve a single policy component
        """
        if "policy" in components:
            return components["policy"]
        if len(components) == 1:
            return next(iter(components.values()))
        raise ValueError(
            "Cannot resolve policy component. Provide a 'policy' key or a single component."
        )

    def _load_normalization_stats(self, stats_path: str) -> None:
        """Load normalization tensors from stats file.

        Args:
            stats_path: Path to pickled stats file
        """
        stats_path = Path(stats_path)
        if not stats_path.exists():
            print(f"Warning: Stats file not found at {stats_path}")
            return

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        # Get device from policy parameters
        try:
            device = next(self._policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Load state normalization stats
        # Handle both flat and nested stats formats
        if "observation.state" in stats:
            state_mean = torch.from_numpy(stats["observation.state"]["mean"]).to(device)
            state_std = torch.from_numpy(stats["observation.state"]["std"]).to(device)
            # Concatenate with current observation stats if present
            if "observation.current" in stats:
                curr_mean = torch.from_numpy(stats["observation.current"]["mean"]).to(device)
                curr_std = torch.from_numpy(stats["observation.current"]["std"]).to(device)
                state_mean = torch.cat([state_mean, curr_mean], dim=-1)
                state_std = torch.cat([state_std, curr_std], dim=-1)
        elif "state" in stats:
            state_mean = torch.from_numpy(stats["state"]["mean"]).to(device)
            state_std = torch.from_numpy(stats["state"]["std"]).to(device)
        else:
            state_mean = torch.zeros(self.state_dim, device=device)
            state_std = torch.ones(self.state_dim, device=device)

        # Load action normalization stats
        if "action" in stats:
            action_mean = torch.from_numpy(stats["action"]["mean"]).to(device)
            action_std = torch.from_numpy(stats["action"]["std"]).to(device)
        else:
            action_mean = torch.zeros(self.action_dim, device=device)
            action_std = torch.ones(self.action_dim, device=device)

        self._state_mean = state_mean
        self._state_std = state_std
        self._action_mean = action_mean
        self._action_std = action_std

    # =========================================================================
    # RTC-required methods
    # =========================================================================

    def encode_memory(
        self, rh: torch.Tensor, cam_images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Encode robot history and camera images into memory for action decoder.

        Args:
            rh: Robot observation history tensor
            cam_images: Camera images tensor

        Returns:
            Dict with encoded memory tensors:
            - 'memory_input': Encoded memory for action decoder
            - 'discrete_semantic_input': Optional discrete input (may be None)
            - 'expert_id': Optional expert identifier (may be None)
        """
        if hasattr(self._policy, "encode_memory"):
            return self._policy.encode_memory(rh, cam_images)
        else:
            raise AttributeError(
                "Underlying policy module does not implement encode_memory(). "
                "Make sure your policy component supports the RTC interface."
            )

    @property
    def body(self) -> nn.Module:
        """Return the action decoder body for guided inference.

        Returns:
            The action decoder module that can be called with
            (time, noise, memory_input, discrete_semantic_input)
        """
        if hasattr(self._policy, "body"):
            return self._policy.body
        elif hasattr(self._policy, "action_decoder"):
            return self._policy.action_decoder
        else:
            raise AttributeError(
                "Underlying policy module does not expose 'body' or 'action_decoder'. "
                "Make sure your policy component supports the RTC interface."
            )

    def freeze_all_model_params(self) -> None:
        """Freeze all parameters (required for VJP in guided inference).

        Sets requires_grad=False for all parameters and switches to eval mode.
        """
        if hasattr(self._policy, "freeze_all_model_params"):
            self._policy.freeze_all_model_params()
        else:
            for param in self._policy.parameters():
                param.requires_grad_(False)
            self._policy.eval()

    @property
    def normalization_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return normalization statistics.

        Returns:
            Tuple of (state_mean, state_std, action_mean, action_std, eps)
        """
        if self._state_mean is None:
            raise ValueError(
                "Normalization stats not loaded. Provide stats_path in policy config."
            )
        return (
            self._state_mean,
            self._state_std,
            self._action_mean,
            self._action_std,
            self.stats_eps,
        )

    # =========================================================================
    # Sequential inference methods
    # =========================================================================

    def predict(self, obs: dict[str, Any]) -> torch.Tensor:
        """Run forward pass for sequential inference.

        Args:
            obs: Observation dict with normalized inputs

        Returns:
            Action tensor from policy
        """
        if hasattr(self._policy, "predict"):
            return self._policy.predict(obs)
        elif hasattr(self._policy, "forward"):
            return self._policy(obs)
        else:
            raise NotImplementedError(
                "Underlying policy module does not implement predict() or forward()."
            )

    def act(self, obs: dict[str, Any]) -> Any:
        """Act method for compatibility.

        Args:
            obs: Observation dict

        Returns:
            Action from policy
        """
        return self.predict(obs)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def to(self, device: torch.device) -> "VFPSingleExpertPolicy":
        """Move policy to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self._policy.to(device)
        if self._state_mean is not None:
            self._state_mean = self._state_mean.to(device)
            self._state_std = self._state_std.to(device)
            self._action_mean = self._action_mean.to(device)
            self._action_std = self._action_std.to(device)
        return self

    def eval(self) -> "VFPSingleExpertPolicy":
        """Set policy to evaluation mode.

        Returns:
            Self for chaining
        """
        self._policy.eval()
        return self

    def train(self, mode: bool = True) -> "VFPSingleExpertPolicy":
        """Set policy training mode.

        Args:
            mode: Training mode flag

        Returns:
            Self for chaining
        """
        self._policy.train(mode)
        return self

    def parameters(self):
        """Return policy parameters iterator."""
        return self._policy.parameters()

    def state_dict(self):
        """Return policy state dict."""
        return self._policy.state_dict()

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load policy state dict.

        Args:
            state_dict: State dictionary
            strict: Whether to enforce strict matching
        """
        return self._policy.load_state_dict(state_dict, strict=strict)
