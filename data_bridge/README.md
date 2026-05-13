# data_bridge

The data-transport layer between the distributed Ray actors. Contains two Ray actors and nothing else:

- **[`ReplayBufferActor`](replay_buffer.py)** — disk-backed (memmap) experience replay with LeRobot-style chunked sampling.
- **[`StateManagerActor`](state_manager.py)** — version-tracked weight broadcasting from trainer to inference loop.

These two actors are created in [run_online_rl.py](../run_online_rl.py) on `training_pc`, with explicit names (`"replay_buffer"` and `"policy_state_manager"`). The trainer rank-0 worker looks them up by name (see [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 421 and 424). The RTC inference loop looks up the state manager by name too ([inference_loop.py:94](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py)). The Sequential actor instead receives the state-manager handle by constructor argument.

Where this fits: see [../docs/02_architecture.md](../docs/02_architecture.md) for the actor graph, and [../docs/06_data_flow.md](../docs/06_data_flow.md) for the episode and weight flows hop-by-hop.

## Table of contents

- [ReplayBufferActor](#replaybufferactor)
  - [Chunking offsets — what gets sampled](#chunking-offsets--what-gets-sampled)
  - [Public API](#public-api)
  - [HIL buffer routing](#hil-buffer-routing)
- [StateManagerActor](#statemanageractor)
  - [Why the version counters](#why-the-version-counters)
  - [Public API](#public-api-1)
- [Files](#files)

## ReplayBufferActor

[`ReplayBufferActor`](replay_buffer.py) wraps [`torchrl.data.TensorDictReplayBuffer`](https://pytorch.org/rl/stable/reference/data.html#tensordictreplaybuffer) with [`LazyMemmapStorage`](https://pytorch.org/rl/stable/reference/data.html#lazymemmapstorage) (disk-backed under `tmp/online_rl_auto_data/`) and a [`SliceSampler`](https://pytorch.org/rl/stable/reference/data.html#slicesampler).

Default constructor parameters (defined in [`replay_buffer.py`](replay_buffer.py)):

| Parameter | Default | Purpose |
|---|---|---|
| `capacity` | `10_000_000` | Maximum timesteps in storage. Memmap is on disk; capacity does not eat RAM. |
| `use_hil_buffer` | `False` | If true, a parallel buffer routes episodes with `control_mode != 0` separately. |
| `proprio_key` | `"proprio"` | Key under which proprio is stored in the incoming TensorDict. |
| `reward_key` | `"reward"` | Key for per-step reward. |
| `action_key` | `"action"` | Key for per-step action. |
| `image_keys` | `("head", "left", "right")` | Camera keys. |
| `action_horizon` | `50` | Future actions per sample. |
| `obs_proprio_history` | `50` | Past proprio steps per sample. |
| `obs_images_history` | `1` | Past image steps per sample (note: in `lerobot_qchunk` mode this offset is currently computed from `obs_proprio_history`, see code). |
| `chunking_mode` | `"lerobot_qchunk"` | Offset scheme; the alternative is `"classic"`. |
| `strict_length` | `True` | Drop episodes shorter than the window. |
| `compile` | `True` | torch.compile the sampler. |

### Chunking offsets — what gets sampled

`_build_offsets()` produces four integer offset tensors (relative to an "anchor" timestep) that determine which past/future steps go into each sample.

For `chunking_mode="lerobot_qchunk"` (the default):

```python
action_offsets  = arange(0, 50)                     # 0..49  — future actions
reward_offsets  = arange(0, 50)                     # 0..49  — future rewards
proprio_offsets = arange(50, -50, -1)               # 50, 49, ..., -49  — broad time range
image_offsets   = arange(50, -1, -50)               # 50, 0           — sparse image samples
```

`_compute_window()` then derives `anchor` and `episode_slice_len` from the min/max across all offsets. With the defaults above, the window is `[-49, +50]` so each sampled trajectory slice is **100 timesteps long** and the anchor is at index 49 inside the slice.

`SliceSampler.slice_len` is set to this computed `episode_slice_len`. `_pack_lerobot_like()` then gathers the appropriate offsets from each `(B, T)` window and returns a `(B,)` shaped TensorDict with these keys:

| Output key | Shape | Source offsets |
|---|---|---|
| `action` | `(B, 50, action_dim)` | `action_offsets` |
| `labels.reward` | `(B, 50)` | `reward_offsets` |
| `labels.done` | `(B, 50)` | `reward_offsets` (aligned) |
| `observation.state` | `(B, 100, state_dim)` | `proprio_offsets` |
| `observation.images.cam_head` | `(B, 2, 3, H, W)` | `image_offsets` |
| `observation.images.cam_left` | same | same |
| `observation.images.cam_right` | same | same |
| `episode` | `(B,)` | scalar at anchor |
| `control_mode` | `(B,)` | scalar at anchor |
| `task_index` | `(B,)` | scalar at anchor |

The previous version of this README documented `slice_len=80` as a constructor parameter. That parameter no longer exists — the window length is computed from offsets. Open [replay_buffer.py](replay_buffer.py)'s `_build_offsets` if you need to change the shapes.

### Public API

| Method | Called by | What it does |
|---|---|---|
| `add(episode_tensordict, separate_key="control_mode")` | The reward labeler, via `replay_buffer.add.remote(td)` | Appends the entire episode (a `(T, ...)`-batched TensorDict) to the memmap. With `use_hil_buffer=True`, routes by `episode_tensordict[separate_key][0]`. |
| `sample(batch_size)` | The trainer's rank-0 worker, via `replay_buffer.sample.remote(B)` | Samples `B` packed LeRobot-shaped windows. With `use_hil_buffer=True`, draws `B/2` from each buffer. |
| `size()` | The trainer's warmup loop, polling until the buffer is full enough | Total timesteps stored. |

### HIL buffer routing

When `use_hil_buffer=True`, the actor creates a second `TensorDictReplayBuffer` under `tmp/online_rl_hil_data/`. `add()` reads `episode_tensordict["control_mode"][0]` and routes:

- `control_mode == 0` → auto buffer.
- `control_mode != 0` (e.g. `1` for HIL-controlled segments) → HIL buffer.

`sample(B)` returns `B/2` from each, concatenated. The default is `use_hil_buffer=False`, which means [run_online_rl.py](../run_online_rl.py) uses only the auto buffer.

## StateManagerActor

[`StateManagerActor`](state_manager.py) is intentionally tiny — it owns one reference and two integers. The reference points to a CPU state dict that the trainer placed in Ray's [Plasma object store](../docs/10_glossary.md#plasma-object-store) via `ray.put(...)`.

### Why the version counters

The actor exposes one `update_state` (write side) and one `get_state` (read side). It maintains:

```python
controller_version = 0    # what the inference loop has already applied
trainer_version    = 0    # what the trainer has already pushed
```

`update_state(ref)` overwrites the reference and increments `trainer_version`. `get_state()` returns the reference iff `controller_version != trainer_version`, then advances `controller_version`. So the same weights are never loaded twice — even if the inference loop polls every episode boundary while the trainer pushes once every several iterations.

The trainer side of this protocol is in [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) lines 532–542. The inference side is in [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) lines 99–109.

### Public API

| Method | Called by | What it does |
|---|---|---|
| `update_state(new_state_ref)` | Trainer rank-0, via `policy_state_manager.update_state.remote(weights_ref)` | Replaces the reference, increments `trainer_version`. Logs `weight pushed to Plasma...`. |
| `get_state()` | Inference loop, between episodes | Returns the reference if versions disagree (and advances `controller_version`), else returns `None`. Logs `received weight from plasma...` on hits. |

The replacement reference is what triggers Plasma's garbage collector to release the previous one. Holding a CPU state dict in Plasma is cheap because Plasma is shared memory, not RPC.

## Files

| File | Purpose |
|---|---|
| [`replay_buffer.py`](replay_buffer.py) | `ReplayBufferActor` — memmap-backed replay buffer with LeRobot-style chunking. |
| [`state_manager.py`](state_manager.py) | `StateManagerActor` — versioned weight reference holder. |
