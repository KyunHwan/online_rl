"""VFP single expert policy stub."""
from __future__ import annotations

from typing import Any

from torch import nn

from env_actor.policy.registry import POLICY_REGISTRY


@POLICY_REGISTRY.register("vfp_single_expert")
class VFPSingleExpertPolicy:
    """Policy wrapper that holds component modules."""

    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None:
        self.components = components
        self.config = kwargs

    def act(self, obs: dict[str, Any]) -> Any:
        raise NotImplementedError("VFPSingleExpertPolicy.act is not implemented yet")
