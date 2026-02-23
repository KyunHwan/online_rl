# env_actor/robot_io_interface

Hardware abstraction layer for communicating with the robot. Provides a uniform API for reading sensor state and publishing actions, regardless of the underlying robot platform.

## Interface / Bridge Pattern

[`controller_interface.py`](controller_interface.py) is the robot-agnostic API. It dispatches to a `ControllerBridge` under `robots/<robot>/`:

```
ControllerInterface(runtime_params, inference_runtime_topics_config, robot="igris_b")
    └── robots/igris_b/controller_bridge.py → ControllerBridge
```

## Key Methods

| Method | Description |
|--------|-------------|
| `start_state_readers()` | Starts background sensor reading (e.g., camera streams, joint encoders) |
| `init_robot_position()` | Moves the robot to a starting pose; returns initial joint positions |
| `read_state()` | Returns the current observation as a dict of numpy arrays |
| `publish_action(action, prev_joint)` | Sends an action to the robot; applies slew-rate limiting |
| `shutdown()` | Cleans up hardware connections |

**Properties:**

| Property | Description |
|----------|-------------|
| `DT` | Control loop timestep in seconds (derived from `HZ` in runtime params) |
| `policy_update_period` | Number of control steps between policy updates |

## Data Formats

**`read_state()` output:**
```python
{
    "proprio": np.ndarray,  # (state_dim,) float32 — joint angles, gripper state, etc.
    "head": np.ndarray,     # (3, H, W) uint8 — head camera image (CHW, RGB)
    "left": np.ndarray,     # (3, H, W) uint8 — left camera image
    "right": np.ndarray,    # (3, H, W) uint8 — right camera image
}
```

**`publish_action()` input:**
- `action`: numpy array of joint targets
- `prev_joint`: previous joint positions (for slew-rate limiting via `max_delta`)

## Files

| File | Purpose |
|------|---------|
| [`controller_interface.py`](controller_interface.py) | Robot-agnostic API |
| [`robots/igris_b/controller_bridge.py`](robots/igris_b/) | igris_b hardware implementation |
| [`robots/igris_c/controller_bridge.py`](robots/igris_c/) | igris_c hardware implementation |

## Adding a New Robot

1. Create `robots/<robot>/controller_bridge.py` with a `ControllerBridge` class.
2. Implement all methods listed above.
3. Add the import branch in `controller_interface.py`.
