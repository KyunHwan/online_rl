# env_actor/policy

Policy protocol, registry, loader, and the two concrete policies. This is where the neural network policy is built from YAML and exposed to the inference loops.

Where this fits: see [../../docs/05_policy_protocol.md](../../docs/05_policy_protocol.md) for the protocol in detail, [../../docs/04_configuration.md](../../docs/04_configuration.md) for the YAML schema, and [../../docs/07_extending.md](../../docs/07_extending.md#recipe-1-add-a-new-policy) for the "add a new policy" recipe.

## Table of contents

- [The Policy Protocol](#the-policy-protocol)
- [Registry](#registry)
- [Loader](#loader)
- [Weight transfer](#weight-transfer)
- [The two production policies](#the-two-production-policies)
  - [OpenPiPolicy](#openpipolicy)
  - [DsrlOpenpiPolicy (default)](#dsrlopenpipolicy-default)
- [Adding a new policy](#adding-a-new-policy)
- [Files](#files)

## The Policy Protocol

Defined in [`templates/policy.py`](templates/policy.py) as a [PEP 544](https://peps.python.org/pep-0544/) `Protocol` with `@runtime_checkable`:

```python
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface): ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed, action_chunk_size): ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

It is structural — your policy class needs the methods, not inheritance. Three contract points:

1. **`self.components = components`** — the inference loop loads weight updates by iterating `policy.components.keys()`. The attribute name is exact.
2. **numpy in, numpy out** — both `predict` and `guided_inference` get numpy, return numpy. Torch lives only inside.
3. **Normalization happens inside the policy** — the `data_normalization_interface` is for the policy to call, not for the surrounding loop.

See [../../docs/08_invariants.md](../../docs/08_invariants.md) for the full invariant list and what breaks if you violate them.

## Registry

[`registry/core.py`](registry/core.py) defines a generic `Registry` class — a simple key-to-class map with optional base-class enforcement. The shared instance for policies is `POLICY_REGISTRY`, imported from [`registry/__init__.py`](registry/__init__.py).

This registry is a **local copy** of the trainer's registry pattern (the trainer has its own, also in [trainer/trainer/registry/](../../trainer/trainer/registry/), but they are independent — no shared state). The reason for the local copy: the env actor must be able to import its policies without instantiating any of the trainer's heavier registries.

Register a class with a decorator:

```python
from env_actor.policy.registry import POLICY_REGISTRY

@POLICY_REGISTRY.register("my_policy")
class MyPolicy:
    ...
```

[`registry/plugins.py`](registry/plugins.py) exists for future plugin-style extensions (lazy `importlib` of registered modules); it is not currently driven from a config but is wired up so that you can call `load_plugins(["my.package"])` if you need to.

## Loader

[`utils/loader.py`](utils/loader.py) — `build_policy(policy_yaml_path)` is the entry point. The RTC inference loop ([../auto/inference_algorithms/rtc/actors/inference_loop.py](../auto/inference_algorithms/rtc/actors/inference_loop.py)) and the Sequential actor ([../auto/inference_algorithms/sequential/sequential_actor.py](../auto/inference_algorithms/sequential/sequential_actor.py)) both call it.

What it does, in order:

1. `load_config(policy_yaml_path)` — uses the **trainer's** config loader (`from trainer.trainer.config.loader import load_config`). The outer repo therefore cannot build a policy without the [trainer submodule](../../docs/10_glossary.md#git-submodule) checked out and importable.
2. Resolve `model.component_config_paths` relative to the YAML's directory. Absolute paths are kept absolute.
3. Build components with `PolicyConstructorModelFactory().build(resolved_paths)` (from `trainer/`). The factory returns `dict[str, nn.Module]` — one per entry in `component_config_paths`.
4. If `checkpoint_path` is set on the policy YAML, load `<component_name>.pt` files from that directory into each module via `module.load_state_dict()`.
5. Look up `policy.type` in `POLICY_REGISTRY`. If absent, auto-import `env_actor.policy.policies.<type>.<type>` to trigger registration.
6. Instantiate the policy class with `policy_cls(components=components, **policy.params)`.

The trainer cross-references: [trainer/docs/04_concepts.md](../../trainer/docs/04_concepts.md) explains the Registry / Factory pattern in depth; [trainer/docs/05_configuration.md](../../trainer/docs/05_configuration.md) documents the schema the component YAMLs use.

## Weight transfer

[`utils/weight_transfer.py`](utils/weight_transfer.py) — `load_state_dict_cpu_into_module(module, sd_cpu, strict=True)` copies a CPU state dict (received from Ray's [Plasma object store](../../docs/10_glossary.md#plasma-object-store)) into a live `nn.Module`. It matches each tensor's `device` and `dtype` to the target module's current state dict so the live module can run in autocast'd `bfloat16` while the trainer pushes `float32` weights — the cast happens at load time, not at every forward pass.

If the live module is wrapped (e.g. in `DDP`), the function unwraps via `module.module if hasattr(module, "module") else module`.

## The two production policies

### OpenPiPolicy

[`policies/openpi_policy/openpi_policy.py`](policies/openpi_policy/openpi_policy.py).

Single-component wrapper around `OpenPiBatchedWrapper` (the PyTorch port of OpenPI). The factory wraps every component in a `GraphModel` (a module-graph container); `OpenPiPolicy._resolve_wrapper()` reaches past `graph_model.graph_modules["openpi_model"]` to get at the underlying flow-matching model.

Inference details:

- Both `predict` and `guided_inference` extract the latest proprio timestep and the latest image frame per camera, batch to size 1, and call `self._wrapper.predict(batched_obs, noise=None)`. `noise=None` lets the wrapper draw its own ODE seed.
- `norm_stats` is exposed as a metadata attribute — `OpenPiBatchedWrapper` carries its own normalization stats and applies them internally; the `data_normalization_interface` argument is accepted but unused.
- `guided_inference` then applies `compute_guided_prefix_weights(...)` and blends with `prev_action`.

YAML ([`policies/openpi_policy/openpi_policy.yaml`](policies/openpi_policy/openpi_policy.yaml)):

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: openpi_policy
```

Use this as a copy-paste template if you have one end-to-end model.

### DsrlOpenpiPolicy (default)

[`policies/dsrl_openpi_policy/dsrl_openpi_policy.py`](policies/dsrl_openpi_policy/dsrl_openpi_policy.py). **This is the default `--policy_yaml` in [run_online_rl.py](../../run_online_rl.py).**

Four named components run as a pipeline:

```text
images ──▶ backbone           ──▶ features
              │                    │
              │  normalized proprio
              │                    ▼
              └──▶ noise_processor ──▶ flat latent
                                      │
                                      ▼
                                 noise_actor ──▶ noise tensor
                                                   │
                                  raw images + raw │ proprio[0]
                                                   ▼
                                              openpi_model ──▶ action chunk
```

- The **DSRL components** (`backbone`, `noise_processor`, `noise_actor`) consume normalized proprio. The `DataNormalizationInterface` is called inside `_run_inference()`.
- The **OpenPI component** consumes **raw** proprio and **raw** images (it manages its own normalization). The DSRL-produced noise tensor is passed in as the OpenPI flow-matching seed instead of `torch.randn`.

The constructor's `params.checkpoint_path` (set on the policy YAML) is where this class loads `backbone.pt`, `noise_processor.pt`, and `noise_actor.pt` from. OpenPI loads its own weights via the `ckpt_dir` field inside the OpenPI component YAML — that is why the policy code explicitly does *not* try to overwrite `openpi_model` weights from `checkpoint_path`.

`warmup()` builds a dummy `obs` dict at the expected shapes (`obs_proprio_history`, `proprio_state_dim`, 240×320 cameras) and calls `_run_inference` once under `torch.inference_mode()` to trigger CUDA kernel selection.

## Adding a new policy

See [../../docs/07_extending.md](../../docs/07_extending.md#recipe-1-add-a-new-policy) for the step-by-step recipe.

## Files

| File | Purpose |
|---|---|
| [`templates/policy.py`](templates/policy.py) | `Policy` Protocol. |
| [`registry/core.py`](registry/core.py) | Generic `Registry` class. |
| [`registry/__init__.py`](registry/__init__.py) | Exports `POLICY_REGISTRY`. |
| [`registry/plugins.py`](registry/plugins.py) | `load_plugins([module_names])` for lazy registry imports. |
| [`utils/loader.py`](utils/loader.py) | `build_policy(yaml_path)` — the entrypoint everything else uses. |
| [`utils/weight_transfer.py`](utils/weight_transfer.py) | `load_state_dict_cpu_into_module(module, sd_cpu, strict)` — used by the inference loop on weight pulls. |
| [`policies/openpi_policy/openpi_policy.py`](policies/openpi_policy/openpi_policy.py) | The simpler single-component OpenPI policy. |
| [`policies/openpi_policy/openpi_policy.yaml`](policies/openpi_policy/openpi_policy.yaml) | Its YAML config. |
| [`policies/dsrl_openpi_policy/dsrl_openpi_policy.py`](policies/dsrl_openpi_policy/dsrl_openpi_policy.py) | Default policy: DSRL noise actor feeding OpenPI. |
| [`policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml`](policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml) | Its YAML config. |
| `policies/{openpi_policy,dsrl_openpi_policy}/components/*.yaml` | Per-component build configs consumed by `PolicyConstructorModelFactory`. |
