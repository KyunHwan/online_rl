# env_actor/policy/registry

A small, self-contained registry the env actor uses to map a YAML's `policy.type` string to a `Policy` class.

This is a **separate registry from the trainer's**. The trainer ([trainer/trainer/registry/](../../../trainer/trainer/registry/)) maintains its own registries for trainers, datasets, optimizers, losses, etc. Those are independent of this one. The env actor only needs a policy registry, hence the lighter local copy.

Where this fits: the registry is consumed by [`build_policy()`](../utils/loader.py); see [../README.md](../README.md#registry) for the full loader story. For the conceptual pattern (registry + factory + plugins), the trainer's [trainer/docs/04_concepts.md](../../../trainer/docs/04_concepts.md) is the canonical explanation.

## Files

| File | Purpose |
|---|---|
| [`core.py`](core.py) | `Registry` generic class with optional base-class enforcement. |
| [`__init__.py`](__init__.py) | Constructs the shared `POLICY_REGISTRY` instance and re-exports it. |
| [`plugins.py`](plugins.py) | `load_plugins(modules)` — lazy `importlib` of plugin module names to trigger registrations. Not currently driven from config but available for use. |

## Public API

```python
from env_actor.policy.registry import POLICY_REGISTRY

@POLICY_REGISTRY.register("my_policy")     # decorator form
class MyPolicy: ...

POLICY_REGISTRY.add("explicit_key", SomePolicy)   # imperative form
cls = POLICY_REGISTRY.get("my_policy")            # lookup, raises KeyError if missing
POLICY_REGISTRY.has("my_policy")                  # bool
POLICY_REGISTRY.keys()                            # list[str]
```

`Registry.__init__` accepts `expected_base` to enforce that registered items are subclasses (or instances) of a given type. The current `POLICY_REGISTRY` does not set one — duck-typing the `Policy` Protocol is enough.

## How `build_policy` finds your class

[`build_policy()`](../utils/loader.py) calls `POLICY_REGISTRY.has(policy_type)`. If your policy is not yet imported (and therefore not registered), the loader tries:

```python
importlib.import_module(f"env_actor.policy.policies.{policy_type}.{policy_type}")
```

That import triggers your `@POLICY_REGISTRY.register("...")` decorator, registers the class, and the loader looks it up. This convention is why the directory layout under `env_actor/policy/policies/` matters — folder name and inner module file name must both match the `policy.type` string.

If you put your policy outside that folder layout, either pre-import it before calling `build_policy`, or extend the loader's auto-import fallback. The simpler option is just to follow the layout.
