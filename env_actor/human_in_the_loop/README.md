# env_actor/human_in_the_loop

Human-guided inference with teleoperation and intervention capabilities. This module mirrors the structure of [`auto/`](../auto/) but adds a human intervention layer: an operator can take over control of the robot in real time, and the system records both autonomous and human-guided actions.

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

[`action_mux/action_mux.py`](action_mux/action_mux.py) — blends policy-predicted actions with teleoperation actions based on the intervention state.

[`action_mux/intervention_switch.py`](action_mux/intervention_switch.py) — tracks whether the human is currently intervening.

[`action_mux/teleop_provider.py`](action_mux/teleop_provider.py) — wraps the teleoperation interface to provide actions in the same format as the policy.

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

| Aspect | `auto/` | `human_in_the_loop/` |
|--------|---------|---------------------|
| Action source | Policy only | Policy + teleoperator (blended) |
| Intervention | None | Pedal-based switch |
| Data recording | `episode_recorder` | Records both auto and human actions |
| Use case | Production online RL | Data collection, safety override |

The core policy, normalization, and inference logic are the same. The HIL layer adds the action mux, teleop, and intervention switching on top.
