# env_actor/policy

Policy loading, registration, protocol definition, and implementations. This is where the neural network policy is built from YAML configuration, registered by name, and exposed to inference loops via a standard protocol.

## Policy Protocol

Defined in [`templates/policy.py`](templates/policy.py) as a Python `Protocol`:

```python
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface): ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed, action_chunk_size): ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

**Key contract:**
- `components` is a `dict[str, nn.Module]` — the model modules built by the factory. The policy must store these as `self.components` so the inference loop can update weights via `policy.components`.
- `predict()` and `guided_inference()` receive a `DataNormalizationInterface` and call it internally. All input data is numpy; all output is numpy.
- `warmup()` runs dummy forward passes to trigger `torch.backends.cudnn.benchmark` kernel selection.

## Registry

[`registry/core.py`](registry/core.py) provides a generic `Registry` class. The policy-specific instance is `POLICY_REGISTRY` (imported from [`registry/__init__.py`](registry/__init__.py)).

Register a policy class with a decorator:

```python
from env_actor.policy.registry import POLICY_REGISTRY

@POLICY_REGISTRY.register("my_policy")
class MyPolicy:
    ...
```

## Policy Loader

[`utils/loader.py`](utils/loader.py) — `build_policy()` is the entry point for constructing a policy from a YAML config:

1. Loads the YAML via the trainer's config loader.
2. Resolves `model.component_config_paths` relative to the YAML directory.
3. Builds components using `PolicyConstructorModelFactory` (from `trainer/`).
4. Looks up `policy.type` in `POLICY_REGISTRY` (auto-imports the module if not yet registered).
5. Instantiates the policy class with the built components.

## Weight Transfer

[`utils/weight_transfer.py`](utils/weight_transfer.py) — `load_state_dict_cpu_into_module()` transfers CPU state dicts (received from Ray's object store) into GPU-resident modules, matching each tensor's device and dtype to the target.

## Implementations

### OpenPiPolicy

[`policies/openpi_policy/openpi_policy.py`](policies/openpi_policy/openpi_policy.py) — wraps an `OpenPiBatchedWrapper` built by the model factory.

**What it does:**
- Resolves the wrapper from the components dict (unwraps `GraphModel` → `graph_modules` → `openpi_model`).
- Exposes metadata: `action_dim`, `action_horizon`, `state_dim`, `norm_stats`.
- `predict()`: extracts the latest timestep from observation history, adds a batch dimension, delegates to the wrapper, removes the batch dimension.
- `guided_inference()`: same as predict, plus computes `compute_guided_prefix_weights()` and blends `prev_action * weights + pred_actions * (1 - weights)`.

**Policy YAML** ([`policies/openpi_policy/openpi_policy.yaml`](policies/openpi_policy/openpi_policy.yaml)):

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: openpi_policy
```

## How to Add a New Policy

1. Create `policies/<your_policy>/<your_policy>.py`.
2. Implement the `Policy` protocol (see above).
3. Decorate with `@POLICY_REGISTRY.register("your_policy")`.
4. Create a YAML config at `policies/<your_policy>/<your_policy>.yaml` with:
   - `model.component_config_paths` mapping component names to their config YAMLs.
   - `policy.type: your_policy`.
5. The loader will auto-import `env_actor.policy.policies.<your_policy>.<your_policy>` if the type is not already registered.

## File Summary

| File | Purpose |
|------|---------|
| [`templates/policy.py`](templates/policy.py) | `Policy` protocol definition |
| [`registry/core.py`](registry/core.py) | Generic `Registry` class |
| [`registry/__init__.py`](registry/__init__.py) | `POLICY_REGISTRY` instance |
| [`utils/loader.py`](utils/loader.py) | `build_policy()` — YAML → components → policy |
| [`utils/weight_transfer.py`](utils/weight_transfer.py) | CPU → GPU state dict transfer |
| [`policies/openpi_policy/openpi_policy.py`](policies/openpi_policy/openpi_policy.py) | OpenPI policy implementation |
| [`policies/openpi_policy/openpi_policy.yaml`](policies/openpi_policy/openpi_policy.yaml) | OpenPI policy config |
