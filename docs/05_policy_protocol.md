# 05 — Policy protocol

This page documents the contract a policy class must satisfy to plug into `online_rl`. After reading it you should be able to write a minimal custom policy, register it, write its YAML, and pass it through `--policy_yaml` without modifying any of the surrounding pipeline.

## Table of contents

- [The `Policy` Protocol](#the-policy-protocol)
- [The `components` contract](#the-components-contract)
- [`predict()` vs `guided_inference()`](#predict-vs-guided_inference)
- [`warmup()`](#warmup)
- [`freeze_all_model_params()`](#freeze_all_model_params)
- [Walkthrough: `OpenPiPolicy`](#walkthrough-openpipolicy)
- [Walkthrough: `DsrlOpenpiPolicy`](#walkthrough-dsrlopenpipolicy)
- [Numpy in, numpy out — the boundary](#numpy-in-numpy-out--the-boundary)

## The `Policy` Protocol

Defined in [env_actor/policy/templates/policy.py](../env_actor/policy/templates/policy.py) as a [PEP 544 Protocol](https://peps.python.org/pep-0544/):

```python
@runtime_checkable
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface): ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed, action_chunk_size): ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

`Policy` is a structural Protocol, not a base class. A class satisfies it by exposing the right methods — no inheritance is required, and you do not need to subclass `nn.Module`. The two production policies (`OpenPiPolicy` and `DsrlOpenpiPolicy`) wrap their `nn.Module` components rather than being modules themselves.

## The `components` contract

Every policy receives a `components: dict[str, nn.Module]` in its constructor. The policy **must** store this dict as `self.components`, because the trainer uses that exact attribute name when broadcasting weights.

The flow:

1. The trainer holds the trainable models under `trainer.models[<name>]` — same names as `policy.components` keys.
2. At checkpoint time the trainer extracts CPU state dicts and pushes them to the `StateManagerActor` via `ray.put` (see [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 539–542).
3. The inference loop calls `policy_state_manager_handle.get_state.remote()`, gets back a dict shaped `{model_name: state_dict}`, then iterates the keys and calls `load_state_dict_cpu_into_module(policy.components[name], state_dict, strict=True)`.

So `components` is the policy's exposed model-weights surface. If your policy uses an `nn.Module` that the trainer should not update online, leave it out of `components` and store it under a different attribute. Conversely, if the trainer updates a model the policy must consume, that model must appear in `components` under the same name as in the trainer's `models` dict.

`build_policy()` ([env_actor/policy/utils/loader.py](../env_actor/policy/utils/loader.py)) constructs the components dict by calling `PolicyConstructorModelFactory().build(resolved_paths)` and passes the result straight into your `__init__`. If the factory returns a single module, the loader wraps it as `{"main": <module>}`.

## `predict()` vs `guided_inference()`

The two inference methods exist because the two algorithms have different needs.

| Method | Called by | Receives | Returns |
|---|---|---|---|
| `predict(obs, norm)` | `SequentialActor` ([env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py)) | `obs` dict from `DataManagerInterface.serve_raw_obs_state()` + normalization handle | `np.ndarray` shape `(action_horizon, action_dim)` float32 |
| `guided_inference(input_data, norm, min_executed, chunk_size)` | RTC inference loop ([env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py)) | `input_data` snapshot from `SharedMemoryInterface.atomic_read_for_inference()` | `np.ndarray` shape `(action_chunk_size, action_dim)` float32 |

The `obs` dict (for `predict`) has these keys:

| Key | Shape | Dtype |
|---|---|---|
| `proprio` | `(proprio_history_size, proprio_state_dim)` | `float32` |
| `head` | `(num_img_obs, 3, H, W)` | `uint8` |
| `left` | same | `uint8` |
| `right` | same | `uint8` |
| `prompt` (optional) | string | — |

The `input_data` dict (for `guided_inference`) is a superset — it has everything `predict` gets plus the RTC bookkeeping fields:

| Extra key | Shape | What it is |
|---|---|---|
| `prev_action` | `(action_chunk_size, action_dim)` | The unexecuted tail of the previous action chunk, padded with zeros at the end. |
| `est_delay` | scalar `int` | Estimated inference latency in control steps (from a sliding-window max over recent measured delays). |
| `num_control_iters` | scalar `int` | How many control steps have been executed since the last `write_action_chunk_n_update_iter_val`. The inference loop passes this back when it writes the new chunk so the counter can be decremented atomically. |
| `action` | `(action_chunk_size, action_dim)` | The current full chunk (mostly useful for inspection — `prev_action` is what you blend with). |

The `guided_inference` body in both production policies follows the same pattern:

```python
pred_actions = self._run_inference(input_data, norm)
weights = compute_guided_prefix_weights(input_data["est_delay"],
                                        min_num_actions_executed,
                                        action_chunk_size,
                                        schedule="exp")
return input_data["prev_action"] * weights[:, None] + pred_actions * (1.0 - weights[:, None])
```

`compute_guided_prefix_weights()` ([env_actor/inference_engine_utils/action_inpainting.py](../env_actor/inference_engine_utils/action_inpainting.py)) returns a `(action_chunk_size,)` float32 array with:

- weights ≈ 1.0 in the first `est_delay` slots — keep the old actions, they have already been committed.
- exponential decay across the middle band.
- weights = 0.0 in the last `min_num_actions_executed` slots — use the new prediction.

This is the [action inpainting](10_glossary.md#action-inpainting) trick from [RTC](10_glossary.md#rtc).

A policy can implement both methods the same way (just call `_run_inference` and skip the blending) if it does not care about the RTC blend. But then RTC will have visible action discontinuities every inference call.

## `warmup()`

Called once at process startup by both inference algorithms. Its job is to trigger any one-time CUDA setup before the realtime loop begins — typically a single dummy forward pass at the expected batch shape so `torch.backends.cudnn.benchmark = True` can pick the fastest kernel.

The inference loop wraps the call in a `try/except` and tolerates failures: `print(f"Warmup encountered error (may be expected for minimal inputs): {e}")`. But a working warmup measurably reduces the first real inference's latency.

`DsrlOpenpiPolicy.warmup()` is a good template: it constructs a dummy `obs` dict with zero arrays at the right shapes (taking the `obs_proprio_history` and `proprio_state_dim` from its constructor, and using a hardcoded 320×240 image size to match the default runtime JSON), and calls `_run_inference` once under `torch.inference_mode()`.

## `freeze_all_model_params()`

Called by the inference loop after `build_policy()` (in practice, only via the policy's own logic — neither the RTC nor the Sequential actor calls it directly today). Setting every `nn.Parameter.requires_grad = False` is cheap insurance; the inference loop also wraps the forward pass in `torch.inference_mode()` so gradients never accumulate even if you skip this.

## Walkthrough: `OpenPiPolicy`

[env_actor/policy/policies/openpi_policy/openpi_policy.py](../env_actor/policy/policies/openpi_policy/openpi_policy.py).

Single-component policy. Its constructor receives `{"openpi_model": <GraphModel>}` and resolves the inner `OpenPiBatchedWrapper` from `graph_model.graph_modules["openpi_model"]`. Why this unwrap step exists: the trainer's `PolicyConstructorModelFactory` wraps every component in a `GraphModel` (a thin module graph with named submodules) to expose a uniform interface; the policy reaches past that wrapper to get at the OpenPI flow-matching model directly.

`predict()` extracts the latest proprio step and the latest image frame per camera, batches them to size 1, calls `self._wrapper.predict(batched_obs, noise=None)`, and unbatches. The `noise=None` argument tells the wrapper to draw its own noise for the flow-matching ODE.

`guided_inference()` does the same forward pass, then applies the inpainting blend.

`norm_stats` is exposed as a metadata attribute — `OpenPiBatchedWrapper` carries its own copy of the normalization stats and applies them internally. The `data_normalization_interface` argument is accepted for protocol compatibility but not used.

This is the simpler policy. Pick it as a copy-paste template if you have a single end-to-end model.

## Walkthrough: `DsrlOpenpiPolicy`

[env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py). This is the **default** policy.

Multi-component policy with four named components, run as a four-stage pipeline:

```text
images (raw)              ┌──── normalized proprio ────┐
   │                      │                            │
   ▼                      ▼                            ▼
backbone(images=...) → noise_processor(data=...) → noise_actor(flat_features=...) → noise
   │                                                                                  │
   │ raw proprio + raw images                                                         │
   └────────────────────────────────────────────────► openpi_model(observation, noise)
                                                              │
                                                              ▼
                                                        action chunk
```

Each component is a `GraphModel` (factory-built), called with the named input the YAML declared. The keys (`backbone`, `noise_processor`, `noise_actor`, `openpi_model`) must match exactly between:

- the policy YAML's `model.component_config_paths`,
- the trainer YAML's `model.component_build_args`,
- the keys the policy code uses (`self.components["backbone"]`, etc.).

If you rename, rename in all three places at once.

The data flow is split:

- The DSRL stages (`backbone`, `noise_processor`, `noise_actor`) consume **normalized** proprio (via the `DataNormalizationInterface`).
- The OpenPI stage consumes **raw** proprio and **raw** images (it has its own normalization inside the model). The DSRL-produced "noise" tensor is passed in as the OpenPI flow-matching seed instead of `torch.randn`.

This is the design from the DSRL+OpenPI experiment — the DSRL actor learns to produce structured noise that the OpenPI denoiser then converts into an action chunk.

## Numpy in, numpy out — the boundary

A policy is the only place torch tensors live in the inference path. Every other component sees numpy:

- Input to the policy: `dict[str, np.ndarray]`.
- Output of the policy: `np.ndarray`.
- `DataNormalizationInterface`: numpy-only. No torch imports.

If you write a policy that returns a torch tensor, the control loop will crash when it tries to `astype(np.float32)` the result. If you make the normalization manager torch-aware, you have broken the [normalization invariant](08_invariants.md#normalization-is-inside-the-policy).

`predict()` and `guided_inference()` both convert numpy → torch internally, do the forward pass, and convert torch → numpy on the way out. Use `torch.from_numpy(arr).float().to(device)` going in and `tensor.cpu().float().numpy()` coming out. The exact lines are visible in both production policies.

Next: [06_data_flow.md](06_data_flow.md) traces one episode's data from sensor read to weight update.
