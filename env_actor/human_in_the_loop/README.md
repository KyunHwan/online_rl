# env_actor/human_in_the_loop

Human-guided inference with teleoperation and intervention capabilities. This module is intended to mirror the structure of [`auto/`](../auto/) but adds a human intervention layer: an operator would take over control of the robot in real time, and the system would record both autonomous and human-guided actions.

## Status

**Not currently wired into [run_online_rl.py](../../run_online_rl.py).** No file in this subtree is imported by the live entrypoint. The directory is a parallel scaffolding for a future HIL-aware entrypoint that does not yet exist. The `--human_reward_labeler` flag only swaps the reward labeler — it does **not** activate any code under this directory. See [../../docs/09_troubleshooting.md](../../docs/09_troubleshooting.md#why-isnt-my-pedalteleop-doing-anything-when-i-pass---human_reward_labeler) for the consequences.

Below is what is **present in the directory**, written in the future tense ("would") since none of it runs today.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ┌──────────────┐     ┌────────────┐                     │
│  │ Policy       │────>│            │    ┌─────────────┐  │
│  │ (inference)  │     │ Action Mux │───>│ Robot I/O   │  │
│  └──────────────┘     │            │    │ (publish)   │  │
│                       │            │    └─────────────┘  │
│  ┌──────────────┐     │            │                     │
│  │ Teleop       │────>│            │                     │
│  │ Provider     │     └────────────┘                     │
│  └──────────────┘          ▲                             │
│                            │                             │
│  ┌──────────────┐          │                             │
│  │ Intervention │──────────┘                             │
│  │ Switch       │ (pedal controls who drives)            │
│  └──────────────┘                                        │
└──────────────────────────────────────────────────────────┘
```

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `action_mux/` | Multiplexes between policy and teleoperator actions |
| `teleoperation/` | Reads human operator input (Dynamixel arms, Manus gloves) |
| `intervention_methods/` | Intervention triggers (e.g., foot pedal) |
| `io_interface/` | Robot I/O for HIL mode (similar to [`robot_io_interface/`](../robot_io_interface/)) |
| `inference_algorithms/` | RTC and Sequential inference, mirroring `auto/` but with HIL integration |

## Key Components

### Action Mux

[`action_mux/action_mux.py`](action_mux/action_mux.py) — would blend policy-predicted actions with teleoperation actions based on the intervention state.

[`action_mux/intervention_switch.py`](action_mux/intervention_switch.py) — would track whether the human is currently intervening.

[`action_mux/teleop_provider.py`](action_mux/teleop_provider.py) — would wrap the teleoperation interface to provide actions in the same format as the policy.

### Teleoperation

`teleoperation/robots/igris_b/` contains:
- [`arms_dynamixel.py`](teleoperation/robots/igris_b/arms_dynamixel.py) — reads arm joint positions from Dynamixel servos used as input devices.
- [`hands_manus.py`](teleoperation/robots/igris_b/hands_manus.py) — reads hand/finger positions from Manus VR gloves.

### Intervention Methods

`intervention_methods/pedal/` — a foot pedal interface:
- [`publisher/pedal_publisher.py`](intervention_methods/pedal/publisher/pedal_publisher.py) — publishes pedal state.
- [`subscriber/pedal_subscriber.py`](intervention_methods/pedal/subscriber/pedal_subscriber.py) — subscribes to pedal events to toggle intervention.

### Inference Algorithms

Mirrors [`auto/inference_algorithms/`](../auto/inference_algorithms/) with the same RTC and Sequential patterns, but the control loop integrates the action mux and teleoperation:

- `inference_algorithms/rtc/` — RTC with human intervention support.
- `inference_algorithms/sequential/` — Sequential with human intervention support.

Each has its own data manager bridges under `data_manager/robots/`.

### I/O Interface

`io_interface/` provides a `ControllerInterface` for HIL mode with robot-specific bridges:
- [`io_interface/controller_interface.py`](io_interface/controller_interface.py)
- `io_interface/robots/igris_b/controller_bridge.py`
- `io_interface/robots/igris_b/utils/camera_utils.py` — camera image processing utilities.
- `io_interface/robots/igris_b/utils/data_dict.py` — observation dictionary construction.

## Comparison with `auto/`

| Aspect | `auto/` (live) | `human_in_the_loop/` (scaffolding) |
|--------|---------|---------------------|
| Action source | Policy only | Would blend policy + teleop |
| Intervention | None | Would use pedal-based switch |
| Data recording | `episode_recorder` | Would record both auto and human actions |
| Status in this repo | Active code path | No live caller; needs new entrypoint |

The core policy, normalization, and inference logic would be shared with `auto/`. The HIL layer would add the action mux, teleop, and intervention switching on top.

## What's missing to make this live

A new entrypoint — e.g. `run_online_rl_hil.py`, **which does not exist in this repo today** — would need to:

1. Import an HIL inference actor from [`inference_algorithms/`](inference_algorithms/) instead of from `env_actor/auto/inference_algorithms/`.
2. Construct the action mux, teleop provider, and pedal subscriber.
3. Pass them into the HIL inference actor.
4. Likely set `use_hil_buffer=True` on `ReplayBufferActor` so HIL-segments are stored separately (see [../../data_bridge/README.md](../../data_bridge/README.md#hil-buffer-routing)).

Until that entrypoint is written, every `.py` under this directory is unreachable from any path that `run_online_rl.py` follows.
