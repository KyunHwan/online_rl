# env_actor/auto/inference_algorithms

The two autonomous inference algorithms that drive the robot. Selected at the command line via `--inference_algorithm`:

| `--inference_algorithm` | Implementation | Subdirectory |
|---|---|---|
| `rtc` (default) | Two-process realtime action chunking. | [`rtc/`](rtc/) |
| `sequential` | Single-process synchronous loop. | [`sequential/`](sequential/) |

Both algorithms produce episodes in the same TensorDict format, push them through the same `episode_queue`, and load weights from the same `StateManagerActor`. The differences are in how often the policy runs, how the control loop is paced, and how the inference and control loops share state.

Where this fits: see [../README.md](../README.md) for the parent `env_actor/auto/` overview, [../../../docs/02_architecture.md](../../../docs/02_architecture.md) for the actor graph, and [../../../docs/06_data_flow.md](../../../docs/06_data_flow.md) for end-to-end episode traces.

## Table of contents

- [RTC (default)](#rtc-default)
  - [Two-process architecture](#two-process-architecture)
  - [Shared-memory layout](#shared-memory-layout)
  - [Synchronization primitives](#synchronization-primitives)
  - [Files (RTC)](#files-rtc)
- [Sequential](#sequential)
  - [Files (Sequential)](#files-sequential)
- [Comparison](#comparison)
- [Adding a new robot](#adding-a-new-robot)

## RTC (default)

[`rtc/rtc_actor.py`](rtc/rtc_actor.py) is a Ray actor (`@ray.remote(num_gpus=1, num_cpus=4)`) pinned to `inference_pc`. Its `start()` method does no inference itself — it allocates a set of shared-memory blocks and spawns two child processes.

### Two-process architecture

```text
                      RTCActor (Ray actor)
                      ──────────────────────
                          allocates SHM
                          mp.get_context("spawn")
                                │
                       ┌────────┴────────┐
                       │                 │
                       ▼                 ▼
        start_inference() proc    start_control() proc
        ──────────────────────    ────────────────────
        builds policy (GPU)       reads robot state
        loops:                    20 Hz loop:
          wait_for_min_actions      read_state()
          atomic_read_for_           atomic_write_obs_
            inference()                and_increment_
          policy.guided_              get_action()
            inference()             publish_action()
          write_action_chunk_       (every 1000 steps)
            n_update_iter_val()     signal_episode_complete
          (between eps:             serve_train_data_buffer
            check StateManager      ray.put + queue.put
            for weights)
```

The two processes:

- **`start_inference`** ([`rtc/actors/inference_loop.py`](rtc/actors/inference_loop.py)) — builds the policy via `build_policy(policy_yaml_path)`, warms it, and runs `guided_inference()` repeatedly. Between episodes it looks up `policy_state_manager` by name and applies any new weights via `load_state_dict_cpu_into_module()`.
- **`start_control`** ([`rtc/actors/control_loop.py`](rtc/actors/control_loop.py)) — runs the 20 Hz control loop. Each step it reads robot state, writes it to shared memory, reads the current action from shared memory, and publishes that action to the robot. At episode end (1000 steps) it serves the recorded episode to the Ray Queue.

The two processes are spawned by `multiprocessing.get_context("spawn")` (not Ray). They communicate exclusively through shared memory and multiprocessing synchronization primitives.

Why two processes (not threads): the GPU forward pass holds the GIL for tens of milliseconds at a time. If both loops shared one Python interpreter, the realtime control loop's 50 ms tick budget would blow on every inference call. See [../../../docs/02_architecture.md](../../../docs/02_architecture.md#the-rtc-two-process-model).

### Shared-memory layout

The parent allocates these via `create_shared_ndarray()` ([`rtc/data_manager/utils/shared_memory_utils.py`](rtc/data_manager/utils/shared_memory_utils.py)). Dimensions come from `RuntimeParams` (the JSON config):

| Key | Shape | Dtype |
|---|---|---|
| `proprio` | `(proprio_history_size, proprio_state_dim)` | `float32` |
| `head` | `(num_img_obs, 3, mono_img_resize_height, mono_img_resize_width)` | `uint8` |
| `left` | same | `uint8` |
| `right` | same | `uint8` |
| `action` | `(action_chunk_size, action_dim)` | `float32` |

Both children attach to these blocks by name via `attach_shared_ndarray()` and access them through [`SharedMemoryInterface`](rtc/data_manager/shm_manager_interface.py), which dispatches to a per-robot bridge ([`rtc/data_manager/robots/igris_b/shm_manager_bridge.py`](rtc/data_manager/robots/igris_b/shm_manager_bridge.py)).

Only the parent process unlinks the blocks (cleanup); children call `resource_tracker.unregister(...)` to opt out of auto-unlink on exit.

### Synchronization primitives

All created from `ctx = mp.get_context("spawn")` in the parent and passed to both children:

| Primitive | Role |
|---|---|
| `RLock` | Mutual exclusion for shared-memory reads/writes. |
| `Condition(lock)` — `control_iter_cond` | Control notifies, inference waits in `wait_for_min_actions`. |
| `Condition(lock)` — `inference_ready_cond` | Inference sets ready, control waits in `wait_for_inference_ready` between episodes. |
| `Event` — `stop_event` | Global shutdown. Set on actor teardown or child-process crash. |
| `Event` — `episode_complete_event` | Control signals episode end; inference reads it via `wait_for_min_actions` returning `'episode_complete'`. |
| `Value('i')` — `num_control_iters` | How many control steps since the last inference. |
| `Value(c_bool)` — `inference_ready_flag` | True while inference is ready to drive a new episode. |

Inference paces itself by `min_num_actions_executed = 35` (hardcoded in [`inference_loop.py`](rtc/actors/inference_loop.py)). With `action_chunk_size = 50`, this leaves ≤15 unexecuted actions for `guided_inference()` to inpaint over.

Control paces itself by `episode_length = 1000` (hardcoded in [`control_loop.py`](rtc/actors/control_loop.py)).

### Files (RTC)

| File | Purpose |
|---|---|
| [`rtc/rtc_actor.py`](rtc/rtc_actor.py) | Ray actor. Allocates SHM, spawns processes, joins, cleans up. |
| [`rtc/actors/inference_loop.py`](rtc/actors/inference_loop.py) | GPU process: build policy, run `guided_inference`, weight updates between episodes. |
| [`rtc/actors/control_loop.py`](rtc/actors/control_loop.py) | CPU process: read robot state, write SHM, publish actions, record episodes. |
| [`rtc/data_manager/shm_manager_interface.py`](rtc/data_manager/shm_manager_interface.py) | Robot-agnostic shared-memory API. |
| [`rtc/data_manager/robots/igris_b/shm_manager_bridge.py`](rtc/data_manager/robots/igris_b/shm_manager_bridge.py) | igris_b SHM bridge. |
| [`rtc/data_manager/utils/shared_memory_utils.py`](rtc/data_manager/utils/shared_memory_utils.py) | `ShmArraySpec`, `create_shared_ndarray`, `attach_shared_ndarray`. |
| [`rtc/data_manager/utils/max_deque.py`](rtc/data_manager/utils/max_deque.py) | Sliding-window max of recent delays, used as `est_delay` in guided inference. |

## Sequential

[`sequential/sequential_actor.py`](sequential/sequential_actor.py) is a single Ray actor (`@ray.remote(num_gpus=1)`). It runs a synchronous control loop:

```python
for t in range(9000):
    obs_data = controller.read_state()
    if t % policy_update_period == 0:
        obs = data_manager.serve_raw_obs_state()
        action_chunk = policy.predict(obs, normalization_iface)   # numpy in, numpy out
        data_manager.buffer_action_chunk(action_chunk, t)
    action = data_manager.get_current_action(t)
    controller.publish_action(action, prev_joint)
    # ... timing
```

No shared memory, no second process. The actor owns its policy directly, its observation history (in `DataManagerBridge`), and its episode recorder.

The `policy_state_manager` handle is received as a **constructor argument** in `SequentialActor.__init__()` — it does **not** call `ray.get_actor("policy_state_manager")`. Weight updates happen between episodes (the outer `while True` loop in `start()` checks `policy_state_manager_handle.get_state.remote()` at the top of every iteration after the first episode).

Inference cadence is controlled by `policy_update_period` from `RuntimeParams` — every N control steps, run `policy.predict()`; between policy calls, execute the cached action chunk by indexing into it.

### Files (Sequential)

| File | Purpose |
|---|---|
| [`sequential/sequential_actor.py`](sequential/sequential_actor.py) | The Ray actor + full control loop. |
| [`sequential/data_manager/data_manager_interface.py`](sequential/data_manager/data_manager_interface.py) | Robot-agnostic in-process data manager. |
| [`sequential/data_manager/robots/igris_b/data_manager_bridge.py`](sequential/data_manager/robots/igris_b/data_manager_bridge.py) | igris_b in-process observation history + action buffering. |
| [`sequential/data_manager/robots/igris_c/data_manager_bridge.py`](sequential/data_manager/robots/igris_c/data_manager_bridge.py) | Stub — raises `NotImplementedError`. |

## Comparison

| Aspect | RTC | Sequential |
|---|---|---|
| Processes | 2 (inference + control) | 1 |
| Synchronization | `multiprocessing.shared_memory` + `RLock`/`Condition`/`Event` | none needed |
| Policy call frequency | When `num_control_iters ≥ 35` (every ~1.75 s at 20 Hz, give or take delay) | Every `policy_update_period` control steps |
| Uses `guided_inference()` / inpainting? | yes | no — calls `predict()` |
| Realtime guarantee | Control loop is GIL-free of inference | Control loop is blocked during each inference |
| StateManager handle acquisition | `ray.get_actor("policy_state_manager")` by name (in inference subprocess) | Constructor argument from [run_online_rl.py](../../../run_online_rl.py) |
| Robustness to renaming the actor | breaks | OK |

## Adding a new robot

Each robot needs both bridges:

| File to add | What it must implement |
|---|---|
| `rtc/data_manager/robots/<robot>/shm_manager_bridge.py` | A `SharedMemoryManager` class matching the methods listed in [`shm_manager_interface.py`](rtc/data_manager/shm_manager_interface.py). |
| `sequential/data_manager/robots/<robot>/data_manager_bridge.py` | A `DataManagerBridge` class matching the methods listed in [`sequential/data_manager/data_manager_interface.py`](sequential/data_manager/data_manager_interface.py). |

And both interfaces' dispatch tables need an `elif robot == "<robot>":` branch. See [../../../docs/07_extending.md](../../../docs/07_extending.md#recipe-2-add-a-new-robot) for the full file list.
