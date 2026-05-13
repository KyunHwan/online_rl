# 04 — Configuration

The system has three independent layers of configuration. Mixing them up is one of the most common cold-start mistakes. This page documents each layer end-to-end: file format, CLI flag, parser, keys actually read, and the source of truth.

## Table of contents

- [The three layers at a glance](#the-three-layers-at-a-glance)
- [Layer 1: Train config YAML](#layer-1-train-config-yaml)
- [Layer 2: Policy config YAML](#layer-2-policy-config-yaml)
- [Layer 3: Runtime JSON (params + topics)](#layer-3-runtime-json-params--topics)
- [A worked example for igris_b](#a-worked-example-for-igris_b)
- [Why `norm_stats_file_path` is a per-deployment edit](#why-norm_stats_file_path-is-a-per-deployment-edit)
- [igris_c — currently a stub](#igris_c--currently-a-stub)

## The three layers at a glance

| Layer | Format | CLI flag | Loader | Source of truth |
|---|---|---|---|---|
| Train config | YAML | `--train_config` | `train_func()` in [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) | [trainer/docs/05_configuration.md](../trainer/docs/05_configuration.md) |
| Policy config | YAML | `--policy_yaml` | `build_policy()` in [env_actor/policy/utils/loader.py](../env_actor/policy/utils/loader.py) | This page + [05_policy_protocol.md](05_policy_protocol.md) |
| Runtime params | JSON | `--inference_runtime_params_config` | `RuntimeParams.__init__()` in [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py) | The JSON file on disk |
| Runtime topics | JSON | `--inference_runtime_topics_config` | Parsed in `controller_loop` / `sequential_actor`; consumed by `GenericRecorder` (in `controller_bridge.py`) | The JSON file on disk |

The CLI flags are defined in the `if __name__ == "__main__"` block of [run_online_rl.py](../run_online_rl.py).

## Layer 1: Train config YAML

**Owns:** the training loop. What model components to build, what dataset to load, which loss/optimizer to use, which checkpoint to resume from, when to save.

**Read by:** the trainer (a [git submodule](10_glossary.md#git-submodule)). The outer entrypoint just hands the file path to `TorchTrainer(train_loop_per_worker=train_func, train_loop_config=train_config_path)`. Inside, `train_func()` reads the YAML with `load_config()` and validates it with `validate_config()` against a Pydantic schema.

**Schema:** documented in [trainer/docs/05_configuration.md](../trainer/docs/05_configuration.md). Do not re-derive it from this page — open the trainer's doc, which is kept current with the schema.

**Default in [run_online_rl.py](../run_online_rl.py):**

```text
/home/user/Projects/online_rl/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml
```

This is an absolute path on the original developer's machine. You must pass `--train_config` explicitly or update the default to point at your local copy.

**Important keys this outer repo depends on:**

- `train.save_dir` — checkpoints land here; the trainer pushes weights to the `StateManagerActor` at every `save_every * 25` iterations.
- `model.component_build_args[<name>].online_update` — controls which model components the trainer broadcasts. Only components with `online_update: true` and `freeze: false` are pushed to `StateManagerActor`. The names must match keys in `policy.components` on the env actor side.

## Layer 2: Policy config YAML

**Owns:** the **inference-time** policy. Which model components to build, which `Policy` class to instantiate, and any policy-specific kwargs.

**Read by:** [build_policy()](../env_actor/policy/utils/loader.py) — called from the RTC inference loop ([env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py)) and the Sequential actor ([env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py)).

**What the loader actually reads** (from [build_policy()](../env_actor/policy/utils/loader.py)):

| Key | Required? | Purpose |
|---|---|---|
| `model.component_config_paths` | yes | Dict mapping component name → component YAML path. Paths are resolved relative to the policy YAML's directory. Each component YAML is parsed by the trainer's `PolicyConstructorModelFactory` which returns an `nn.Module`. |
| `policy.type` | yes | Registry key. Looked up in `POLICY_REGISTRY`. If missing, the loader tries `importlib.import_module("env_actor.policy.policies.<type>.<type>")` to trigger registration. |
| `policy.params` | no | Extra kwargs passed to the policy constructor. |
| `checkpoint_path` | no | Top-level directory containing per-component `.pt` weight files. Each file is loaded with `module.load_state_dict()`. Naming: `<component_name>.pt`. |

**Default in [run_online_rl.py](../run_online_rl.py):**

```text
./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml
```

The other supported policy is `./env_actor/policy/policies/openpi_policy/openpi_policy.yaml`.

**Worked example — [openpi_policy.yaml](../env_actor/policy/policies/openpi_policy/openpi_policy.yaml):**

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: openpi_policy
```

The path `components/openpi_batched.yaml` is resolved relative to the YAML's directory, so it expands to `env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml`. `openpi_policy` is the registry key on [`OpenPiPolicy`](../env_actor/policy/policies/openpi_policy/openpi_policy.py).

**Worked example — [dsrl_openpi_policy.yaml](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml):**

```yaml
model:
  component_config_paths:
    backbone:        components/backbone.yaml
    noise_processor: components/noise_processor.yaml
    noise_actor:     components/noise_actor.yaml
    openpi_model:    components/openpi_model.yaml

policy:
  type: dsrl_openpi_policy
  params:
    checkpoint_path: /home/robros/Projects/online_rl/env_actor/policy/policies/dsrl_openpi_policy/checkpoints/exp1/epoch_10
    obs_proprio_history: 50
```

Four components are built and passed into [`DsrlOpenpiPolicy.__init__()`](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py). The `params.checkpoint_path` is consumed by the policy class itself (not by the loader's top-level `checkpoint_path` handling) to load the DSRL weights `backbone.pt`, `noise_processor.pt`, `noise_actor.pt`. The OpenPI weights are loaded by the OpenPI component YAML's own `ckpt_dir` key, not from this checkpoint directory.

The component YAMLs themselves follow the trainer's component config schema — see [trainer/docs/05_configuration.md](../trainer/docs/05_configuration.md). Examples live in [env_actor/policy/policies/dsrl_openpi_policy/components/](../env_actor/policy/policies/dsrl_openpi_policy/components/) and [env_actor/policy/policies/openpi_policy/components/](../env_actor/policy/policies/openpi_policy/components/).

## Layer 3: Runtime JSON (params + topics)

**Owns:** robot hardware specifics — control frequency, state dimensions, image sizes, action chunk size, ROS topic names, the path to the normalization-stats pickle.

**Read by:**

- `inference_runtime_params.json` → parsed in `RTCActor.start()` and `SequentialActor.__init__()`, then wrapped in `RuntimeParams` (the class defined in [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py)).
- `inference_runtime_topics.json` → parsed similarly; passed to `ControllerInterface`, which forwards it to `GenericRecorder` inside the robot bridge.

**Every property `RuntimeParams` exposes** (verified against [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py)):

| Property | JSON key | Type | What it controls |
|---|---|---|---|
| `HZ` | `HZ` | int | Control loop frequency (Hz). `DT = 1/HZ`. |
| `policy_update_period` | `policy_update_period` | int | Used by the Sequential actor: it runs `policy.predict()` every N control steps. Unused by RTC. |
| `max_delta` | `max_delta_deg` | float (deg→rad) | Per-step slew-rate limit, in radians. |
| `proprio_state_dim` | `proprio_state_dim` | int | Width of the proprio vector. Drives the shape of `proprio` shared memory and the policy's expected input. |
| `proprio_history_size` | `proprio_history_size` | int | How many past proprio steps to stack. |
| `camera_names` | `camera_names` | list[str] | Camera identifiers used as dict keys throughout the pipeline. |
| `num_img_obs` | `num_img_obs` | int | How many image frames to stack per camera. |
| `img_obs_every` | `img_obs_every` | int | Subsampling rate for images. |
| `mono_img_resize_width`, `mono_img_resize_height` | `mono_image_resize.width`, `.height` | int, int | Resize target for camera frames after capture. |
| `action_dim` | `action_dim` | int | Width of one action vector. |
| `action_chunk_size` | `action_chunk_size` | int | Length of the predicted action chunk. |
| (private) | `norm_stats_file_path` | string | Absolute path to the pickled normalization stats. Read via `read_stats_file()`. |

`read_stats_file()` opens the pickle file at `norm_stats_file_path` and returns a dict shaped:

```python
{
    "observation.state":   {"mean": np.ndarray, "std": np.ndarray},
    "observation.current": {"mean": np.ndarray, "std": np.ndarray},
    "action":              {"mean": np.ndarray, "std": np.ndarray},
}
```

Consumed by [DataNormalizationBridge.normalize_state()](../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py).

**Topics JSON** ([inference_runtime_topics.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json)): a `topics` dict mapping logical names (`finger`, `finger_current`, `joints`) to ROS2 topic strings, message types, and field slicing rules used by `GenericRecorder` to build the proprio dict. See the JSON for the exact structure.

## A worked example for igris_b

The pieces a working `igris_b` run uses:

| Piece | Path |
|---|---|
| Robot | `--robot igris_b` |
| Train YAML | `trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml` (under the [trainer submodule](../trainer/)) |
| Policy YAML | [env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml) |
| Runtime params JSON | [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) |
| Runtime topics JSON | [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json) |
| Normalization stats pickle | path inside the runtime JSON's `norm_stats_file_path` — must exist on the inference machine |

The current igris_b JSON values (treat the file as the source of truth, not this table):

| Key | Value |
|---|---|
| `HZ` | `20` |
| `policy_update_period` | `50` |
| `max_delta_deg` | `5` |
| `proprio_state_dim` | `24` |
| `proprio_history_size` | `50` |
| `camera_names` | `["head", "left", "right"]` |
| `num_img_obs` | `1` |
| `img_obs_every` | `1` |
| `mono_image_resize` | `{ width: 320, height: 240 }` |
| `action_dim` | `24` |
| `action_chunk_size` | `50` |
| `norm_stats_file_path` | `/home/robros/Projects/inference_engine/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dataset_stats.pkl` |

## Why `norm_stats_file_path` is a per-deployment edit

The pickle file is produced by the trainer during dataset preparation. The current JSON points at a path on the original developer's box (`/home/robros/Projects/inference_engine/...`) — that file does not exist on yours. `RuntimeParams.read_stats_file()` prints `File not found at: <path>` and returns `None`, which then crashes the first `DataNormalizationBridge.normalize_state()` call with `TypeError: 'NoneType' object is not subscriptable`.

Three options to fix:

1. Copy the pickle from wherever your trainer produced it, and edit the JSON to point at the new location.
2. Edit the JSON to point straight at the trainer's saved-stats path on your machine.
3. Generate the stats from your own dataset following the trainer's data-preparation flow.

This is the single most common first-run failure. Both [01_getting_started.md](01_getting_started.md) and [09_troubleshooting.md](09_troubleshooting.md) call it out.

## igris_c — currently a stub

`--robot igris_c` is in the CLI choices but does not work today. The `igris_c` runtime config directory only contains `init_params.py` — there is no `inference_runtime_params.py`, no `inference_runtime_params.json`, no `inference_runtime_topics.json`. The import at [run_online_rl.py:92](../run_online_rl.py) crashes immediately:

```python
from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams
# ModuleNotFoundError
```

Treat `igris_c` as a scaffolded port that nobody has finished. See [07_extending.md](07_extending.md#recipe-2-add-a-new-robot) for the file list that would be needed to complete it.

Next: [05_policy_protocol.md](05_policy_protocol.md) explains the Policy interface that a custom policy class must satisfy.
