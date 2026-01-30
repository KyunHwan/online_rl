"""Global registries for policy components."""
from __future__ import annotations

from env_actor.policy.templates.trainer import Trainer
from trainer.registry.core import Registry

from typing import Any

POLICY_REGISTRY: Registry[type[Trainer]] = Registry("policy", expected_base=Trainer)