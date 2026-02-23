# env_actor/episode_recorder

Records observation-action pairs during episodes for use as training data. The recorded data is packaged into TensorDicts and pushed to the episode queue for reward labeling.

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
| [`episode_recorder_interface.py`](episode_recorder_interface.py) | Robot-agnostic interface |
| [`robots/igris_b/episode_recorder_bridge.py`](robots/igris_b/episode_recorder_bridge.py) | igris_b recording implementation |
| [`robots/igris_c/episode_recorder_bridge.py`](robots/igris_c/episode_recorder_bridge.py) | igris_c recording implementation |

## Adding a New Robot

1. Create `robots/<robot>/episode_recorder_bridge.py` with an `EpisodeRecorderBridge` class.
2. Implement `init_train_data_buffer()`, `add_obs_state()`, `add_action()`, and `serve_train_data_buffer()`.
3. Add the import branch in `episode_recorder_interface.py`.
