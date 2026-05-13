# env_actor/robot_io_interface/robots/igris_c

Robot bridge for the IGRIS_C platform.

## Status

> **Partial stub.** [`controller_bridge.py`](controller_bridge.py) exists as a skeleton (~80 lines) where every method raises `NotImplementedError`. The directory tree (`utils/`, `__init__.py`) is in place. The matching runtime config files are **missing** (see [../../../runtime_settings_configs/README.md](../../../runtime_settings_configs/README.md#igris_c-status)). Today, `--robot igris_c` crashes at the import of `RuntimeParams` in [run_online_rl.py:92](../../../../run_online_rl.py), before reaching this bridge.

## Files

| File | What's there |
|---|---|
| [`controller_bridge.py`](controller_bridge.py) | Skeleton class — `__init__` and every method raise `NotImplementedError`. Useful as a template for the required interface (matches [`igris_b/controller_bridge.py`](../igris_b/controller_bridge.py) method-by-method). |
| [`utils/__init__.py`](utils/__init__.py) | Empty placeholder. The igris_b version under [`../igris_b/utils/`](../igris_b/utils/) holds `camera_utils.py` and `data_dict.py` — the same will be needed here. |
| [`__init__.py`](__init__.py) | Empty package marker. |

## Required interface

The stub `ControllerBridge` must implement, with the same signatures as in [`../igris_b/controller_bridge.py`](../igris_b/controller_bridge.py):

| Method | Returns |
|---|---|
| `__init__(runtime_params, inference_runtime_topics_config)` | — |
| `read_state()` | `dict[str, np.ndarray]` with keys `proprio`, `head`, `left`, `right` |
| `publish_action(action, prev_joint)` | `(smoothed_joints, fingers)` numpy arrays |
| `start_state_readers()` | None |
| `init_robot_position()` | numpy array of joint positions |
| `recorder_rate_controller()` | rate-controller object (currently only used by Sequential) |
| `shutdown()` | None |
| `DT` (property) | float seconds = `1/HZ` |
| `policy_update_period` (property) | int |

## Steps to make `--robot igris_c` actually run

In rough order:

1. Add the missing runtime-config files (`inference_runtime_params.py`, `inference_runtime_params.json`, `inference_runtime_topics.json`) under [`../../../runtime_settings_configs/robots/igris_c/`](../../../runtime_settings_configs/robots/igris_c/). The JSON values must match igris_c's actual hardware.
2. Fill in `init_params.py` with `INIT_JOINT_LIST`, `INIT_HAND_LIST`, and `IGRIS_C_STATE_KEYS`.
3. Implement [`controller_bridge.py`](controller_bridge.py) here — ROS2 publishers/subscribers, camera capture, slew-rate limiting, etc.
4. Implement the matching bridges in [`../../../nom_stats_manager/`](../../../nom_stats_manager/), [`../../../episode_recorder/`](../../../episode_recorder/), [`../../../auto/inference_algorithms/rtc/data_manager/`](../../../auto/inference_algorithms/rtc/data_manager/), and [`../../../auto/inference_algorithms/sequential/data_manager/`](../../../auto/inference_algorithms/sequential/data_manager/) — see [../../../../docs/07_extending.md](../../../../docs/07_extending.md#recipe-2-add-a-new-robot).
5. Add `elif robot == "igris_c":` import branches in each interface class.
6. Add `"igris_c"` to argparse `--robot` choices in [run_online_rl.py](../../../../run_online_rl.py) (already present, but verify).

Use [`../igris_b/`](../igris_b/) as the reference. The interfaces dispatch by string so the bridge's method names must match exactly.
