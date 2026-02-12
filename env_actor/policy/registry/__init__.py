"""Global registries for policy components."""
from __future__ import annotations

from env_actor.policy.templates.policy import Policy
from trainer.trainer.registry.core import Registry

from typing import Any

POLICY_REGISTRY: Registry[type[Policy]] = Registry("policy", expected_base=Policy)
