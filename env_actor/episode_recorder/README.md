# env_actor/episode_recorder

Records observation-action pairs during episodes for use as training data. The recorded data is packaged into [TensorDicts](../../docs/10_glossary.md#tensordict) and pushed to the episode queue for reward labeling.

Where this fits: the control loop calls `add_obs_state` and `add_action` per step, then `serve_train_data_buffer(episode_id)` at episode end. The returned list of sub-episode TensorDicts is then `ray.put`'d into Plasma and enqueued on `episode_queue`. See [../../docs/06_data_flow.md](../../docs/06_data_flow.md#hop-6-episode-end--tensordict--ray-queue).

## Interface / Bridge Pattern

[`episode_recorder_interface.py`](episode_recorder_interface.py) dispatches to robot-specific bridges:

```
EpisodeRecorderInterface(robot="igris_b")
    └── robots/igris_b/episode_recorder_bridge.py → EpisodeRecorderBridge
```

## Key Methods

| Method | Description |
|--------|-------------|
| `init_train_data_buffer()` | Resets internal buffers for a new episode |
| `add_obs_state(obs_data)` | Appends one timestep of observations to the buffer |
| `add_action(action, **kwargs)` | Appends one timestep of actions to the buffer |
| `serve_train_data_buffer(episode_id)` | Packages the buffered data into TensorDicts and returns them |

## Data Flow

```
Control loop                     Episode Recorder              Ray Queue
    │                                │                             │
    ├─ init_train_data_buffer() ────>│                             │
    │                                │                             │
    ├─ add_obs_state(obs) ──────────>│  (accumulate in buffer)     │
    ├─ add_action(action) ──────────>│                             │
    │   ... (repeat per timestep)    │                             │
    │                                │                             │
    ├─ serve_train_data_buffer() ───>│──── TensorDict ───────────>│
    │                                │                             │
```

The control loop (in [`auto/inference_algorithms/rtc/actors/control_loop.py`](../auto/inference_algorithms/rtc/actors/control_loop.py)) calls `add_obs_state()` and `add_action()` at every control step. At the end of an episode, `serve_train_data_buffer()` returns one or more sub-episodes as TensorDicts, which are placed in the Ray object store and enqueued for the reward labeler.

## Files

| File | Purpose |
|------|---------|
| [`episode_recorder_interface.py`](episode_recorder_interface.py) | Robot-agnostic interface; dispatches by string. |
| [`robots/igris_b/episode_recorder_bridge.py`](robots/igris_b/episode_recorder_bridge.py) | igris_b recording. Splits the timeseries into runs of equal `control_mode` (`torch.unique_consecutive`), producing one sub-episode TensorDict per run with derived episode id `base_id * 10000 + segment_idx`. |
| [`robots/igris_c/episode_recorder_bridge.py`](robots/igris_c/episode_recorder_bridge.py) | Empty file — stub. |

## Known doc/code drift

The TensorDict produced by `serve_train_data_buffer` includes `task_index` (integer) but not `task` (string). The auto reward labeler expects a `"task"` string key — see [../../data_labeler/README.md#known-issues](../../data_labeler/README.md#known-issues) and [../../docs/09_troubleshooting.md#auto-labeler-cannot-find-task](../../docs/09_troubleshooting.md#auto-labeler-cannot-find-task).

## Adding a New Robot

1. Create `robots/<robot>/episode_recorder_bridge.py` with an `EpisodeRecorderBridge` class.
2. Implement `init_train_data_buffer()`, `add_obs_state()`, `add_action()`, and `serve_train_data_buffer()`.
3. Add the import branch in `episode_recorder_interface.py`.
