# env_actor/runtime_settings_configs

Robot-specific runtime configuration. Two file types per robot:

- **`inference_runtime_params.py`** — defines a `RuntimeParams` class that wraps a config dict.
- **`inference_runtime_params.json`** — the actual values. **This file is the source of truth.**
- **`inference_runtime_topics.json`** — ROS2 topic names and field-slicing rules.
- **`init_params.py`** — initial joint positions and the proprio-state key list.

The JSON files are what change between deployments. The Python is loaded by [run_online_rl.py](../../run_online_rl.py) and the RTC/Sequential actors.

Where this fits: see [../../docs/04_configuration.md](../../docs/04_configuration.md) for the three-layer config story, [../../docs/06_data_flow.md](../../docs/06_data_flow.md) for how these dimensions flow through the system.

## Table of contents

- [Layout](#layout)
- [What `RuntimeParams` exposes](#what-runtimeparams-exposes)
- [Current igris_b values](#current-igris_b-values)
- [Deployment-specific values you must edit](#deployment-specific-values-you-must-edit)
- [igris_c status](#igris_c-status)
- [Files](#files)

## Layout

```text
runtime_settings_configs/
├── robots/
│   ├── igris_b/
│   │   ├── inference_runtime_params.py     # RuntimeParams class
│   │   ├── inference_runtime_params.json   # actual values  ← source of truth
│   │   ├── inference_runtime_topics.json   # ROS topic config
│   │   └── init_params.py                  # INIT_JOINT, IGRIS_B_STATE_KEYS, ...
│   └── igris_c/
│       ├── init_params.py                  # placeholder, has TODOs
│       └── __init__.py
└── __init__.py
```

## What `RuntimeParams` exposes

Defined in [`robots/igris_b/inference_runtime_params.py`](robots/igris_b/inference_runtime_params.py). Every property is read once from the JSON dict at `__init__` and exposed as a Python attribute:

| Property | JSON key | Type | Used by |
|---|---|---|---|
| `HZ` | `HZ` | int | Control-loop frequency; `DT = 1/HZ` |
| `policy_update_period` | `policy_update_period` | int | Sequential algorithm only — `t % N == 0` triggers a policy call |
| `max_delta` | `max_delta_deg` | float (deg→rad) | `ControllerBridge.publish_action()` clips per-joint motion |
| `proprio_state_dim` | `proprio_state_dim` | int | Width of `proprio` SHM array |
| `proprio_history_size` | `proprio_history_size` | int | Length of proprio FIFO buffer |
| `camera_names` | `camera_names` | list[str] | Camera dict keys used everywhere downstream |
| `num_img_obs` | `num_img_obs` | int | Image-stack depth |
| `img_obs_every` | `img_obs_every` | int | Image-subsample stride |
| `mono_img_resize_width`, `mono_img_resize_height` | `mono_image_resize.width`, `.height` | int | Camera resize target |
| `action_dim` | `action_dim` | int | Width of one action vector |
| `action_chunk_size` | `action_chunk_size` | int | Length of one predicted chunk |
| (used internally) | `norm_stats_file_path` | str | Absolute path to the pickled normalization stats |

`RuntimeParams.read_stats_file()` reads `norm_stats_file_path`, returns the unpickled dict if the file exists, prints `File not found at: <path>` and returns `None` otherwise.

## Current igris_b values

Treat [`robots/igris_b/inference_runtime_params.json`](robots/igris_b/inference_runtime_params.json) as the source of truth, not this table. As of the version committed:

```json
{
  "HZ": 20,
  "max_delta_deg": 5,
  "policy_update_period": 50,
  "mono_image_resize": { "width": 320, "height": 240 },
  "camera_names": ["head", "left", "right"],
  "proprio_state_dim": 24,
  "action_dim": 24,
  "action_chunk_size": 50,
  "proprio_history_size": 50,
  "num_img_obs": 1,
  "img_obs_every": 1,
  "norm_stats_file_path": "/home/robros/Projects/inference_engine/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dataset_stats.pkl"
}
```

[`robots/igris_b/inference_runtime_topics.json`](robots/igris_b/inference_runtime_topics.json) defines:

- `robot_id: "packy"` — used as a topic-prefix variable in `ControllerBridge`.
- `HZ: 20`.
- `topics`: a dict of logical names (`finger`, `finger_current`, `joints`) mapping to `(topic, msg_type, fields)`. The `fields` slice rule tells the `GenericRecorder` how to extract the relevant subset from each incoming message.

[`robots/igris_b/init_params.py`](robots/igris_b/init_params.py) defines:

- `INIT_JOINT_LIST` — 12 floats (right then left arm) in degrees.
- `INIT_HAND_LIST` — 12 floats (right then left hand finger targets).
- `INIT_JOINT` — numpy array of the joint list converted to radians.
- `IGRIS_B_STATE_KEYS` — the ordered list of `/observation/...` keys whose values get concatenated into the proprio vector.

## Deployment-specific values you must edit

Three values in `inference_runtime_params.json` are hard-coded to the original developer's environment:

| Value | Why it's wrong on your box | Fix |
|---|---|---|
| `norm_stats_file_path` | Absolute path on developer's machine | Point at your local `dataset_stats.pkl` |

(The other JSON values are robot-physical and need no edits unless you change hardware.)

Without the fix, the first inference call crashes with `TypeError: 'NoneType' object is not subscriptable` from inside `DataNormalizationBridge.normalize_state`. See [../../docs/09_troubleshooting.md](../../docs/09_troubleshooting.md#norm_stats_file_path-file-not-found).

## igris_c status

The `igris_c` runtime config directory has only `__init__.py` and `init_params.py` (the latter is full of `TODO` placeholders). Missing files that the entrypoint would need:

- `inference_runtime_params.py` (the `RuntimeParams` class)
- `inference_runtime_params.json`
- `inference_runtime_topics.json`

[run_online_rl.py:92](../../run_online_rl.py) tries to import `RuntimeParams` from this missing module — `--robot igris_c` therefore crashes with `ModuleNotFoundError` before any actor starts. See [../../docs/07_extending.md](../../docs/07_extending.md#igris_c--what-is-already-there-what-is-missing).

## Files

| File | Purpose |
|---|---|
| [`robots/igris_b/inference_runtime_params.py`](robots/igris_b/inference_runtime_params.py) | `RuntimeParams` class — parses the JSON dict. |
| [`robots/igris_b/inference_runtime_params.json`](robots/igris_b/inference_runtime_params.json) | igris_b runtime values. |
| [`robots/igris_b/inference_runtime_topics.json`](robots/igris_b/inference_runtime_topics.json) | igris_b ROS2 topic configuration. |
| [`robots/igris_b/init_params.py`](robots/igris_b/init_params.py) | igris_b initial pose and state-key list. |
| [`robots/igris_c/init_params.py`](robots/igris_c/init_params.py) | igris_c initial pose — TODO placeholders only. |
