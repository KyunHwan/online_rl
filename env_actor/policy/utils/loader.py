"""Policy configuration loader and builder for env_actor."""
from __future__ import annotations

import importlib
import os
from typing import Any

import torch

from trainer.trainer.config.loader import load_config
from trainer.trainer.modeling.factories import PolicyConstructorModelFactory

from env_actor.policy.registry import POLICY_REGISTRY
from env_actor.policy.templates.policy import Policy


def load_policy_config(path: str) -> dict[str, Any]:
    """Load a policy YAML config with defaults composition."""
    return load_config(path)


def _resolve_component_paths(
    component_paths: dict[str, str], *, policy_yaml_path: str
) -> dict[str, str]:
    """Resolve component paths relative to the policy YAML directory."""
    base_dir = os.path.dirname(os.path.abspath(policy_yaml_path))
    resolved: dict[str, str] = {}
    for name, rel_path in component_paths.items():
        if os.path.isabs(rel_path):
            resolved[name] = rel_path
        else:
            resolved[name] = os.path.abspath(os.path.join(base_dir, rel_path))
    return resolved


def build_policy(
    policy_yaml_path: str,
    *,
    map_location: str | torch.device | None = "cpu",
) -> Policy:
    """Build a policy from a YAML config path."""
    config = load_policy_config(policy_yaml_path)

    model_cfg = config.get("model", {})
    component_paths = model_cfg.get("component_config_paths", {})
    if not isinstance(component_paths, dict) or not component_paths:
        raise ValueError("model.component_config_paths must be a non-empty mapping")

    resolved_paths = _resolve_component_paths(
        component_paths, policy_yaml_path=policy_yaml_path
    )

    factory = PolicyConstructorModelFactory()
    components = factory.build(resolved_paths)
    if not isinstance(components, dict):
        components = {"main": components}

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        for name, module in components.items():
            ckpt_path = os.path.join(checkpoint_path, f"{name}.pt")
            state = torch.load(ckpt_path, map_location=map_location)
            module.load_state_dict(state)

    policy_spec = config.get("policy", {})
    policy_type = policy_spec.get("type")
    if not policy_type:
        raise ValueError("policy.type must be set in policy config")

    if not POLICY_REGISTRY.has(policy_type):
        module_path = f"env_actor.policy.policies.{policy_type}.{policy_type}"
        importlib.import_module(module_path)

    policy_cls = POLICY_REGISTRY.get(policy_type)
    policy_params = policy_spec.get("params", {})
    return policy_cls(components=components, **policy_params)
