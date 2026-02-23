# data_bridge

Data transport layer between the distributed actors in the online RL pipeline. Contains two Ray actors:

1. **ReplayBufferActor** — disk-backed experience replay buffer.
2. **StateManagerActor** — version-tracked policy weight distribution.

## ReplayBufferActor

**File:** [`replay_buffer.py`](replay_buffer.py)

A Ray actor wrapping `TensorDictReplayBuffer` from [torchrl](https://pytorch.org/rl/) with `LazyMemmapStorage` for disk-backed storage.

### How It Works

- **Storage:** Data is memory-mapped to disk under `tmp/online_rl_auto_data/`. Writing is instant (memmap); no serialization overhead.
- **Sampling:** `SliceSampler` extracts trajectory slices of `slice_len` timesteps (default 80) using the `"episode"` key to identify trajectory boundaries.
- **Capacity:** Default 10 million timesteps.

### Key Methods

| Method | Description |
|--------|-------------|
| `add(episode_tensordict)` | Writes a TensorDict episode to disk via memmap. Returns `True` on success. |
| `sample(batch_size)` | Samples `batch_size` trajectory slices from disk. Returns `TensorDict` shaped `(batch_size, slice_len)`. |
| `size()` | Returns total number of stored timesteps. |

### HIL Buffer Separation

When `use_hil_buffer=True`, a second replay buffer is created under `tmp/online_rl_hil_data/`. Episodes are routed based on the `control_mode` field:
- `control_mode == 0` → auto buffer
- `control_mode == 1` → HIL buffer

Sampling draws equally from both buffers.

### Data Flow

```
EnvActor → ep_queue → RewardLabeler → replay_buffer.add(td) → disk (memmap)
                                                                    │
Trainer ← replay_buffer.sample(batch_size) ←────────────────────────┘
```

## StateManagerActor

**File:** [`state_manager.py`](state_manager.py)

A lightweight Ray actor for distributing updated policy weights from the trainer to the inference actor. Uses Ray's Plasma object store for zero-copy weight transfer.

### How It Works

The trainer pushes a reference to the new state dict into the Plasma store, then calls `update_state()` with that reference. The inference actor periodically calls `get_state()` to check for updates.

Version tracking prevents redundant transfers:
- `trainer_version` increments on each `update_state()`.
- `controller_version` tracks what the inference actor has received.
- `get_state()` returns weights only when versions differ; returns `None` otherwise.

### Key Methods

| Method | Called by | Description |
|--------|----------|-------------|
| `update_state(new_state_ref)` | Trainer | Stores a new weight reference, increments trainer version |
| `get_state()` | Inference actor | Returns weight reference if updated, `None` otherwise |

### Weight Update Flow

```
Trainer                    StateManager                 Inference Actor
   │                           │                              │
   ├─ ray.put(weights) ──────>│                              │
   ├─ update_state(ref) ─────>│ trainer_version++            │
   │                           │                              │
   │                           │<──── get_state() ────────────┤
   │                           │───── ref (if version diff) ─>│
   │                           │      controller_version++    │
   │                           │                              ├─ load weights
```

## Files

| File | Purpose |
|------|---------|
| [`replay_buffer.py`](replay_buffer.py) | `ReplayBufferActor` — disk-backed replay buffer |
| [`state_manager.py`](state_manager.py) | `StateManagerActor` — weight versioning and distribution |
