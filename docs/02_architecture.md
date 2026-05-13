# 02 — Architecture

This page is a complete tour of the Ray actor graph that [run_online_rl.py](../run_online_rl.py) creates. After reading it you should be able to draw the cluster on a whiteboard from memory and explain what each actor does, where it lives, and how data flows between them.

## Table of contents

- [The four actors](#the-four-actors)
- [The full diagram](#the-full-diagram)
- [Actor: ReplayBufferActor](#actor-replaybufferactor)
- [Actor: StateManagerActor](#actor-statemanageractor)
- [Actor: AutoRewardLabelerActor (or ManualRewardLabelerActor)](#actor-autorewardlabeleractor-or-manualrewardlabeleractor)
- [Actor: RTCActor / SequentialActor](#actor-rtcactor--sequentialactor)
- [The RTC two-process model](#the-rtc-two-process-model)
- [Shared memory layout (RTC only)](#shared-memory-layout-rtc-only)
- [RTC synchronization primitives](#rtc-synchronization-primitives)
- [Episode lifecycle (RTC)](#episode-lifecycle-rtc)
- [What is NOT in this diagram](#what-is-not-in-this-diagram)

## The four actors

[run_online_rl.py](../run_online_rl.py)'s `start_online_rl()` function spawns exactly these actors (plus one Ray task and one Ray Queue):

| Actor | Defined in | Resource | Count | Named? |
|---|---|---|---|---|
| `StateManagerActor` | [data_bridge/state_manager.py](../data_bridge/state_manager.py) | `training_pc:1` | 1 | yes, `policy_state_manager` |
| `ReplayBufferActor` | [data_bridge/replay_buffer.py](../data_bridge/replay_buffer.py) | `training_pc:1` | 1 | yes, `replay_buffer` |
| `RTCActor` (default) **or** `SequentialActor` | [env_actor/auto/inference_algorithms/rtc/rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) **or** [env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py) | `inference_pc:1` (RTC: `num_gpus=1, num_cpus=4`; Sequential: `num_gpus=1, num_cpus=4`) | 1 | no |
| `AutoRewardLabelerActor` | [data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py) | `labeling_pc:1, num_gpus=1` | `--num_labeler_gpus` (default 4) | yes, `reward_labeler_<i>` |
| `ManualRewardLabelerActor` (if `--human_reward_labeler`) | [data_labeler/human_in_the_loop/hil_reward_labeler.py](../data_labeler/human_in_the_loop/hil_reward_labeler.py) | `labeling_pc:1` | 1 | no |

Two extra pieces:

- `episode_queue` — a `ray.util.queue.Queue` (Ray Queue, `maxsize=15`) created in the driver. It is the producer/consumer channel between the env actor and the labeler.
- `run_training` — a `@ray.remote` task (not actor) on `training_pc:1` that calls `TorchTrainer.fit(train_loop_per_worker=train_func, scaling_config=ScalingConfig(num_workers=4, use_gpu=True))`. This `TorchTrainer` then launches 4 DDP worker processes inside the `training_pc` allocation. The function it runs, `train_func`, lives in [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py).

## The full diagram

```text
                       ┌─────────────────────────────┐
                       │   labeling_pc (HEAD NODE)   │
                       │                             │
                       │  ┌───────────────────────┐  │
                       │  │ AutoRewardLabeler × N │  │
   ┌───────────────┐   │  │  (Robometer VLM)      │  │
   │  inference_pc │   │  │  ray.queue.get()      │  │
   │               │   │  └───────────┬───────────┘  │
   │ ┌───────────┐ │   │              │              │
   │ │ RTCActor  │ │   └──────────────┼──────────────┘
   │ │  spawns:  │ │                  │  add.remote(td)
   │ │ ┌───────┐ │ │  ray.put(td)     ▼
   │ │ │ infer │ │ │  episode_queue     ┌─────────────────────────────────┐
   │ │ │ loop  │◀┼─┼─────────────────── │  training_pc                    │
   │ │ └───┬───┘ │ │                    │                                 │
   │ │     │ shm │ │   ┌─────────────┐  │  ┌──────────────────────────┐   │
   │ │ ┌───▼───┐ │ │   │  Replay     │◀─┼──│ Trainer (Ray DDP × 4)    │   │
   │ │ │ ctrl  │─┼─┼──▶│  Buffer     │  │  │  train_func()            │   │
   │ │ │ loop  │ │ │   │  (memmap)   │──┼─▶│  sample(batch_size)      │   │
   │ │ └───┬───┘ │ │   └─────────────┘  │  │  TorchTrainer.fit()      │   │
   │ │     │     │ │                    │  └──────────┬───────────────┘   │
   │ └─────┼─────┘ │                    │             │ ray.put(weights)  │
   │       │       │                    │             ▼                   │
   │   robot I/O   │   weights ref      │     ┌───────────────────┐       │
   └───────┼───────┘◀───────────────────┼─────│ StateManagerActor │       │
           ▼                            │     │ "policy_state_..."│       │
       physical robot                   │     └───────────────────┘       │
                                        └─────────────────────────────────┘
```

Three machines, four actor types (five with the manual labeler), one Ray Queue, one named pull point for weights.

## Actor: ReplayBufferActor

[data_bridge/replay_buffer.py](../data_bridge/replay_buffer.py).

A Ray actor wrapping a `TensorDictReplayBuffer` from [torchrl](10_glossary.md#torchrl), backed by `LazyMemmapStorage` on disk under `tmp/online_rl_auto_data/`. Writing is instant (memmap maps the buffer to a file), so the labeler does not block the env actor.

It samples with a `SliceSampler` configured for [LeRobot-style](10_glossary.md#lerobot) chunking. The chunking offsets (built by `_build_offsets()`) determine which past proprioception steps, future action steps, and image timesteps each sample contains. With the defaults (`action_horizon=50`, `obs_proprio_history=50`, `obs_images_history=1`, `chunking_mode="lerobot_qchunk"`), the offsets cover the range `[-49, +50]` in time, so each sampled window is 100 steps long and the anchor is offset 49.

The output of `sample(batch_size)` is a `TensorDict` shaped `[batch_size]` where each element holds:

- `observation.state` — proprio history, `(batch_size, len(proprio_offsets), proprio_state_dim)`.
- `observation.images.cam_head`, `cam_left`, `cam_right` — image frames at the image offsets.
- `action` — future action chunk, `(batch_size, action_horizon, action_dim)`.
- `labels.reward`, `labels.done`, `episode`, `task_index`, `control_mode` — scalar/horizon-aligned metadata.

The HIL buffer split (`use_hil_buffer=True`) routes episodes by `control_mode`: rows with `control_mode==0` go to the auto buffer, others to a separate HIL buffer. Default is off — [run_online_rl.py](../run_online_rl.py) creates `ReplayBufferActor()` with no args.

See [data_bridge/README.md](../data_bridge/README.md) for the full API table.

## Actor: StateManagerActor

[data_bridge/state_manager.py](../data_bridge/state_manager.py).

Tiny actor (one Python file, ~35 lines). It holds a single reference (to a CPU state dict that lives in Ray's [Plasma object store](10_glossary.md#plasma-object-store)) and two integer version counters:

```python
controller_version  # what the inference loop has applied
trainer_version     # what the trainer has pushed
```

The trainer calls `update_state(new_state_ref)` which sets the ref and increments `trainer_version`. The inference loop calls `get_state()`; it returns the ref iff `controller_version != trainer_version` and then advances `controller_version`. So the inference loop only loads weights when something new is available — see [06_data_flow.md](06_data_flow.md#weight-update-half-loop) for the full sequence.

The actor is created with `name="policy_state_manager"`. Both halves of the system look it up:

- The trainer (`train_func` in [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py), line 424) calls `ray.get_actor("policy_state_manager")`.
- The RTC inference loop ([env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py), line 94) calls `ray.get_actor("policy_state_manager")` too.
- The Sequential actor instead receives the handle as a constructor argument from [run_online_rl.py](../run_online_rl.py).

If you rename the named actor in [run_online_rl.py](../run_online_rl.py), RTC and the trainer silently break — the Sequential path keeps working because it gets the handle by reference.

## Actor: AutoRewardLabelerActor (or ManualRewardLabelerActor)

[data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py) / [data_labeler/human_in_the_loop/hil_reward_labeler.py](../data_labeler/human_in_the_loop/hil_reward_labeler.py).

The labeler is the consumer side of the `episode_queue`. Its `start()` method runs a `while True` loop that:

1. `self.episode_queue_handle.get(block=True)` — pulls one `TensorDict` (technically an `ObjectRef[TensorDict]` produced by `ray.put`, which the labeler then `ray.get`s).
2. Annotates the `reward` field. The auto labeler runs the [Robometer](10_glossary.md#robometer) VLM on subsampled frames; the manual labeler shows the frames in a Qt slider and waits for keystrokes.
3. `self.replay_buffer_actor.add.remote(labeled_td)` — pushes the labeled episode to the replay buffer.

`--num_labeler_gpus` controls how many auto-labeler workers spawn in parallel. Each actor requests `num_gpus=1` in its `.options()`, so the cluster must have that many GPUs on the `labeling_pc` node. The auto branch of [run_online_rl.py](../run_online_rl.py) spawns them in a `for` loop and calls `.start.remote()` on each.

The auto labeler reads `episode_data["task"]` from each incoming TensorDict to get the task description string. It does not accept a task-descriptions kwarg — see [09_troubleshooting.md](09_troubleshooting.md#auto-labeler-cannot-find-task) if the labeler crashes with `ValueError`.

The manual labeler is created in [run_online_rl.py](../run_online_rl.py), but `labeler.start.remote()` is **not** called for it (only the auto branch calls `.start.remote()`). This is a known issue — see [09_troubleshooting.md](09_troubleshooting.md#manual-reward-labeler-never-pumps-the-queue).

## Actor: RTCActor / SequentialActor

Exactly one of these spawns based on `--inference_algorithm`:

- **`rtc`** (default) → [env_actor/auto/inference_algorithms/rtc/rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py). See next section.
- **`sequential`** → [env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py). A single Ray actor that runs the read-state → infer → publish loop synchronously, no shared memory.

Sequential is the simpler reference. RTC is what production uses.

## The RTC two-process model

[RTCActor](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) is a Ray actor whose `start()` method does almost no work itself — it allocates a set of `multiprocessing.shared_memory.SharedMemory` blocks, then spawns two child Python processes via `multiprocessing.get_context("spawn")`:

```text
RTCActor (Ray actor on inference_pc)
   │
   ├── ctx.Process(target=start_inference, ...)   # GPU process
   │     • builds policy with build_policy()
   │     • runs guided_inference() in a loop
   │     • writes new action chunks to SHM["action"]
   │     • polls policy_state_manager for weight updates
   │
   └── ctx.Process(target=start_control, ...)     # CPU process
         • reads robot state at 20 Hz (HZ from runtime JSON)
         • writes proprio + camera into SHM
         • reads current action from SHM["action"]
         • publishes action via ControllerInterface.publish_action()
         • records episode → episode_queue at end
```

The reason for two processes (rather than two threads) is the [GIL](10_glossary.md#gil). If both loops shared one Python interpreter, GPU inference would block the realtime 20 Hz control loop. Separate processes mean the kernel scheduler keeps each on its own core.

The Ray actor itself is the parent: it owns the shared memory's lifecycle (it `.close()`s and `.unlink()`s every block in its `finally` clause) and waits on `join()` for both children.

Both children re-initialize Ray inside their process with `ray.init(address="auto", namespace="online_rl")` because Ray's actor-handle objects do not survive `fork`/`spawn` cleanly. The parent `cloudpickle.dumps()` the `episode_queue` handle and passes it to the control loop as bytes; the control loop `cloudpickle.loads()` it after re-initializing Ray.

## Shared memory layout (RTC only)

Allocated in [RTCActor.start()](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) via `create_shared_ndarray()` (see [env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py)):

| Key | Shape | Dtype | Source of dimensions |
|---|---|---|---|
| `proprio` | `(proprio_history_size, proprio_state_dim)` | `float32` | `RuntimeParams` |
| `head` | `(num_img_obs, 3, mono_img_resize_height, mono_img_resize_width)` | `uint8` | `RuntimeParams` |
| `left` | same as `head` | `uint8` | `RuntimeParams` |
| `right` | same as `head` | `uint8` | `RuntimeParams` |
| `action` | `(action_chunk_size, action_dim)` | `float32` | `RuntimeParams` |

For `igris_b` with the default JSON: `proprio` is `(50, 24)`, the three camera arrays are each `(1, 3, 240, 320)`, and `action` is `(50, 24)`.

Each block is created in the parent; the children attach by name via `attach_shared_ndarray()`. The parent is the sole unlinker (cleanup). Children call `resource_tracker.unregister(...)` to opt out of auto-unlink on exit.

## RTC synchronization primitives

All created from the parent `ctx = multiprocessing.get_context("spawn")` and passed into both children:

| Primitive | Purpose | Used by |
|---|---|---|
| `RLock` | Atomic shared-memory reads/writes | both |
| `Condition(lock)` — `control_iter_cond` | Notifies inference that another control step happened | control notifies, inference waits |
| `Condition(lock)` — `inference_ready_cond` | Inference signals it has loaded the policy and is ready for an episode | inference sets, control waits |
| `Event` — `stop_event` | Global shutdown | both |
| `Event` — `episode_complete_event` | End of an episode | control sets, inference waits |
| `Value('i')` — `num_control_iters` | Counter of control steps since the last `write_action_chunk_n_update_iter_val` | control increments, inference reads |
| `Value(c_bool)` — `inference_ready_flag` | Boolean "inference is ready" | inference sets, control waits |

The state-machine logic for these is encapsulated in [shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py), which delegates to a per-robot bridge (`igris_b`).

## Episode lifecycle (RTC)

The sequence as actually executed by the two loops. Inference is the gating side; control is the timed side.

```text
Inference                           Control
─────────                           ───────
build_policy() + warmup()
load any pending weights
set_inference_ready()       ─────▶
                                    wait_for_inference_ready()
                                    clear_episode_complete()
                                    init_robot_position()
                                    bootstrap_obs_history()
                                    init_action_chunk()
                                    ── for t in range(1000): ──
                                        read_state()
                                        atomic_write_obs_and_increment_get_action()
                                        publish_action()
                                        sleep(DT - elapsed)
wait_for_min_actions(35)            ◀── 35 control steps elapsed
set_inference_not_ready()
atomic_read_for_inference()
guided_inference(...)
write_action_chunk_n_update_iter_val()
                                    (control keeps publishing)
... (loop)                          ... (loop until t==1000)
                                    signal_episode_complete()
wait_for_min_actions → 'episode_complete'
                                    serve_train_data_buffer(episode)
                                    episode_queue.put(ray.put(sub_episode))
                                    init_train_data_buffer()
load any pending weights
set_inference_ready()       ─────▶  next episode...
```

`min_num_actions_executed = 35` is hardcoded inside [inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) — the inference loop waits for at least 35 control steps to elapse before running the next inference. With `action_chunk_size = 50`, this leaves ≤15 "fresh" actions in the chunk for `guided_inference()` to inpaint over.

`episode_length = 1000` is hardcoded inside [control_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py).

## What is NOT in this diagram

This is the live data path. The following directories exist in the repo but are **not** wired into [run_online_rl.py](../run_online_rl.py):

- [env_actor/human_in_the_loop/](../env_actor/human_in_the_loop/) — a parallel HIL implementation with teleoperation, action mux, and pedal-based intervention. None of these files are imported by [run_online_rl.py](../run_online_rl.py). They are scaffolding for a future HIL entrypoint that does not yet exist. The `--human_reward_labeler` flag switches the reward labeler only; it does not wire in any of these files. See the Status callout in [env_actor/human_in_the_loop/README.md](../env_actor/human_in_the_loop/README.md).

- All `igris_c` files — the runtime params Python module, the inference runtime JSONs, and several bridges. The CLI accepts `--robot igris_c` but the import at [run_online_rl.py](../run_online_rl.py) line 92 crashes immediately because the `igris_c` runtime-config module does not exist.

Next: [03_distributed_setup.md](03_distributed_setup.md) covers Tailscale, hostnames, and resource pinning.
