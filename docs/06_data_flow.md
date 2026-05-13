# 06 — Data flow

This page traces the life of a single timestep of data from the moment the robot's joint encoders are read to the moment a parameter update lands back on the inference machine. Every hop names the file, the function, the data type, and the transport.

The path is the same for RTC and Sequential except where noted. The default — and what production uses — is RTC.

## Table of contents

- [The end-to-end picture](#the-end-to-end-picture)
- [Hop 1: Robot sensor → numpy state dict](#hop-1-robot-sensor--numpy-state-dict)
- [Hop 2: numpy state → shared memory (RTC) or in-memory history (Sequential)](#hop-2-numpy-state--shared-memory-rtc-or-in-memory-history-sequential)
- [Hop 3: SHM/buffer → numpy snapshot for inference](#hop-3-shmbuffer--numpy-snapshot-for-inference)
- [Hop 4: numpy → policy.predict / policy.guided_inference → numpy action chunk](#hop-4-numpy--policypredict--policyguided_inference--numpy-action-chunk)
- [Hop 5: action chunk → robot](#hop-5-action-chunk--robot)
- [Hop 6: Episode end → TensorDict → Ray Queue](#hop-6-episode-end--tensordict--ray-queue)
- [Hop 7: Ray Queue → labeler → replay buffer](#hop-7-ray-queue--labeler--replay-buffer)
- [Hop 8: Replay buffer → trainer batch](#hop-8-replay-buffer--trainer-batch)
- [Weight-update half-loop](#weight-update-half-loop)
- [The named-actor coupling](#the-named-actor-coupling)

## The end-to-end picture

```text
                                                          ┌─────────────────────────────────┐
                                                          │ trainer (Ray Train DDP × 4)      │
                                                          │                                  │
                                                          │ sample(B) ──▶ train_step() ──▶   │
                                                          │ unwrap_model().state_dict() ──▶  │
                                                          │ ray.put(weights) ──▶            │
                                                          │ state_manager.update_state(ref) │
                                                          └────────────┬─────────────────────┘
                                                                       │
                                                                       │
   robot sensor                                                ┌───────▼────────┐
        │ (1)                                                  │ StateManager   │
        ▼                                                      │ controller_v.  │
   ControllerBridge.read_state()                               │ trainer_v.     │
        │ numpy state dict                                     └──┬─────────────┘
        ▼ (2)                                                     │ (10) get_state.remote()
   SharedMemory["proprio"]/["head"]/["left"]/["right"]            │
        │                                                         ▼
        ▼ (3)                                                  load_state_dict_cpu_into_module(...)
   atomic_read_for_inference()                                    │
        │ numpy snapshot                                          │
        ▼ (4)                                                     │
   policy.guided_inference(...)                                   │
        │ numpy action chunk                                      │
        ▼ (5)                                                     │
   SharedMemory["action"]                                         │
        │                                                         │
        ▼ control loop reads next action                          │
   ControllerBridge.publish_action()                              │
        │                                                         │
        ▼ (end of 1000 steps)                                     │
   EpisodeRecorder.serve_train_data_buffer()                      │
        │ TensorDict                                              │
        ▼ (6) ray.put + queue.put                                 │
   episode_queue (RayQueue)                                       │
        │                                                         │
        ▼ (7)                                                     │
   AutoRewardLabeler.process_episode()                            │
        │ labeled TensorDict                                      │
        ▼ replay_buffer.add.remote(td)                            │
   ReplayBufferActor (memmap on disk)                             │
        │                                                         │
        ▼ (8) sample(batch_size)                                  │
   trainer batches ──────────────────────────────────────────────┘
```

Numbered hops below.

## Hop 1: Robot sensor → numpy state dict

- **File:** [env_actor/robot_io_interface/robots/igris_b/controller_bridge.py](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py)
- **Function:** `ControllerBridge.read_state()`
- **Transport:** ROS2 subscriptions (proprio) + V4L2 (cameras), both kept hot by `start_state_readers()`.
- **Data type:** `dict[str, np.ndarray]`. Keys: `proprio`, `head`, `left`, `right`.
- **Shapes:**
  - `proprio` → `(proprio_state_dim,)` = `(24,)`, float32.
  - Each camera → `(3, mono_img_resize_height, mono_img_resize_width)` = `(3, 240, 320)`, uint8. Note CHW layout — the bridge does `np.transpose(cam_image, (2,0,1))` after the OpenCV BGR resize.

`read_state()` is called once per control step (every `DT = 1/HZ` seconds, i.e. every 50 ms at 20 Hz).

## Hop 2: numpy state → shared memory (RTC) or in-memory history (Sequential)

**RTC path:**

- **File:** [env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)
- **Function:** `SharedMemoryManager.atomic_write_obs_and_increment_get_action(obs, action_chunk_size)`
- **Transport:** [`multiprocessing.shared_memory.SharedMemory`](10_glossary.md#sharedmemory). The proprio history is FIFO-shifted (`[1:] = [:-1]`, new at index 0). Cameras are overwritten in place.
- **Synchronization:** `with self._lock` (an `RLock` shared with the inference process). The same call increments `num_control_iters` and returns the action at index `num_control_iters - 1` from the action chunk.

After the write, `notify_step()` wakes any inference process waiting on `control_iter_cond`.

**Sequential path:**

- **File:** [env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py](../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py)
- **Function:** `DataManagerBridge.update_state_history(obs_data)`
- **Transport:** plain in-process numpy buffers (no shared memory, no lock needed — one process).

## Hop 3: SHM/buffer → numpy snapshot for inference

**RTC path:**

- **File:** [env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)
- **Function:** `SharedMemoryManager.atomic_read_for_inference()`
- **Transport:** `with self._lock` snapshot. Every shared array is `.copy()`-ed into a fresh dict, so the inference process never holds the lock across the GPU forward pass.
- **Output:** `dict[str, np.ndarray]` with the same keys as Hop 2 plus `prev_action` (the unexecuted tail of the previous chunk, zero-padded), `est_delay` (max of the `MaxDeque`), and `num_control_iters`.

The inference loop has been blocked on `wait_for_min_actions(35)` until either 35 control steps have elapsed since the last inference, or the episode ended, or a stop event fired.

**Sequential path:**

- **File:** [env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py](../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py)
- **Function:** `DataManagerBridge.serve_raw_obs_state()`
- **Output:** numpy dict with just the observation keys. No `prev_action`, no `est_delay` — Sequential calls `policy.predict()` rather than `guided_inference()`.

## Hop 4: numpy → policy.predict / policy.guided_inference → numpy action chunk

- **File:** [env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py) (default) or [env_actor/policy/policies/openpi_policy/openpi_policy.py](../env_actor/policy/policies/openpi_policy/openpi_policy.py)
- **Functions:** `predict(obs, norm)` (Sequential) or `guided_inference(input_data, norm, min_executed, chunk_size)` (RTC)
- **Wrapped in:** `with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):`
- **Data type:** numpy goes in, numpy comes out. Torch lives only inside.
- **Output:** `np.ndarray` shape `(action_chunk_size, action_dim)` = `(50, 24)`, float32.

The detailed sequence inside `DsrlOpenpiPolicy` is documented in [05_policy_protocol.md](05_policy_protocol.md#walkthrough-dsrlopenpipolicy).

## Hop 5: action chunk → robot

**RTC path:** the inference loop writes the new chunk back to shared memory:

- **File:** [env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)
- **Function:** `SharedMemoryManager.write_action_chunk_n_update_iter_val(action_chunk, executed)`
- **Effect:** `np.copyto(self._shm_array_dict['action'], action_chunk)`, then `num_control_iters -= executed`, then push current delay to the `MaxDeque`.

The control loop, every step, reads the action at `num_control_iters - 1` and passes it to:

- **File:** [env_actor/robot_io_interface/robots/igris_b/controller_bridge.py](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py)
- **Function:** `ControllerBridge.publish_action(action, prev_joint)`
- **Effect:** Splits the 24-D action into left/right arm joints and finger targets, applies per-joint slew-rate clipping (`max_delta = deg2rad(5)`), publishes via `JointState` and `Float32MultiArray` topics, returns the smoothed joints for the next call's `prev_joint`.

**Sequential path:** same `publish_action()`, called inline in `SequentialActor.start()`.

## Hop 6: Episode end → TensorDict → Ray Queue

After `episode_length = 1000` control steps:

- **File:** [env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py](../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py)
- **Function:** `EpisodeRecorderBridge.serve_train_data_buffer(episode_id)`
- **What happens:**
  1. Stack the per-step `add_obs_state` and `add_action` TensorDicts into a `(T, ...)`-batched TensorDict.
  2. Stamp `episode` id, zero `reward`, set `task_index = 2`, set `("next", "done")[-1] = True`.
  3. Split the timeseries by **runs of equal `control_mode`** using `torch.unique_consecutive`. Each run becomes a sub-episode with its own derived episode id (`base_id * 10000 + segment_idx`). The split exists to keep autonomous segments separate from any future HIL segments.
  4. Return a list of sub-episode TensorDicts.

The control loop does:

```python
for sub_ep in sub_eps:
    sub_ep_data_ref = ray.put(sub_ep)        # → Plasma object store
    episode_queue_handle.put(sub_ep_data_ref, block=True)
```

The queue is bounded at 15 entries. If the labeler falls behind, the env actor blocks on `put`, which protects against unbounded memory growth.

## Hop 7: Ray Queue → labeler → replay buffer

- **File:** [data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py)
- **Functions:** `start()` (the loop) → `process_episode()` (per item)
- **What happens:**
  1. `episode_data_ref = self.episode_queue_handle.get(block=True)` — pulls one entry. `RayQueue.get` auto-dereferences `ObjectRef` so `episode_data` is the TensorDict.
  2. Subsample 32 frames uniformly, run the Robometer VLM, interpolate the per-frame progress scores back to `T` length.
  3. `episode_data[self.reward_key] = torch.from_numpy(progress_scores)`.
  4. `ray.get(self.replay_buffer_actor.add.remote(episode_data))` — blocks until the disk write finishes.

If the manual labeler is selected with `--human_reward_labeler`, the same shape applies but a `QTimer` polls the queue every 100 ms and the human labels frames via the slider.

The replay buffer (`add()` in [data_bridge/replay_buffer.py](../data_bridge/replay_buffer.py)) calls `TensorDictReplayBuffer.extend(td)`. With `LazyMemmapStorage`, "extend" copies the tensors into the on-disk memmap and frees the Python objects immediately. No serialization, no RAM growth.

## Hop 8: Replay buffer → trainer batch

- **File:** [data_bridge/replay_buffer.py](../data_bridge/replay_buffer.py)
- **Function:** `ReplayBufferActor.sample(batch_size)`
- **What happens:** internally calls `SliceSampler.sample(batch_size * episode_slice_len)`, reshapes to `(batch_size, T)`, then `_pack_lerobot_like()` gathers slices for `action`, `labels.reward`, `observation.state`, `observation.images.cam_<name>`, etc. at the configured offsets (see [02_architecture.md](02_architecture.md#actor-replaybufferactor)).
- **Returned to:** the trainer DDP workers via Ray actor RPC. Each worker calls `replay_buffer.sample.remote(B)` and processes the batch.

For the trainer-side of this hop, jump to [trainer/docs/03_ray_online_training.md](../trainer/docs/03_ray_online_training.md).

## Weight-update half-loop

Every `(iterations + 1) % (save_every * 25) == 0` (see [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 526), the trainer (rank-0 worker only) does:

```python
policy_components_weights = {}
for model_name in trainer.models.keys():
    if not config.model.component_build_args[model_name]['freeze'] \
       and config.model.component_build_args[model_name]['online_update']:
        raw_model = unwrap_model(trainer.models[model_name])
        policy_components_weights[model_name] = {
            k: v.cpu() for k, v in raw_model.state_dict().items()
        }
weights_ref = ray.put(policy_components_weights)
policy_state_manager.update_state.remote(weights_ref)
```

The CPU state dict is what reaches `StateManagerActor.update_state(new_state_ref)`. It increments `trainer_version`.

Meanwhile, on the inference machine, the inference loop wakes from `wait_for_min_actions` between episodes and calls:

```python
current_weights = ray.get(policy_state_manager_handle.get_state.remote())
if current_weights is not None:
    for model_name in current_weights.keys():
        if model_name in policy.components.keys():
            load_state_dict_cpu_into_module(
                policy.components[model_name],
                current_weights[model_name],
                strict=True,
            )
```

`get_state()` returns the weights ref iff `controller_version != trainer_version`, and atomically advances `controller_version`. So weights are loaded **at most once per push** and the load happens **only between episodes**, never mid-episode.

[load_state_dict_cpu_into_module()](../env_actor/policy/utils/weight_transfer.py) walks the CPU state dict and copies each tensor `non_blocking=True` to whichever device + dtype the live module currently holds. This is what enables `bfloat16` autocast on the inference side while the trainer pushes float32 weights.

## The named-actor coupling

`StateManagerActor` and `ReplayBufferActor` are created in [run_online_rl.py](../run_online_rl.py) with explicit names (`name="policy_state_manager"`, `name="replay_buffer"`). Three different processes look them up by these names:

| Caller | File | Line |
|---|---|---|
| RTC inference loop | [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) | `ray.get_actor("policy_state_manager")` |
| Trainer rank-0 worker | [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) | `ray.get_actor("replay_buffer")` and `ray.get_actor("policy_state_manager")` |
| Sequential actor | [env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py) | **does not** — it receives `policy_state_manager_handle` as a constructor argument from `run_online_rl.py` |

So renaming either of the two named actors in [run_online_rl.py](../run_online_rl.py) silently breaks the trainer and the RTC inference loop, while the Sequential actor keeps working. If you change the names, change them in all three places.

The replay buffer is found by name only on the trainer side; the labeler receives the actor handle as a constructor argument, so the labeler does not care about the name.

Next: [07_extending.md](07_extending.md) shows how to plug new components into the pipeline.
