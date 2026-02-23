# env_actor

Environment-side logic for the online RL pipeline. This directory handles everything that happens on the inference machine: running policy inference, communicating with robot hardware, recording episodes, and managing data normalization.

## Data Flow

```
Robot Hardware
     │ read_state()
     ▼
┌──────────────┐      ┌────────────────────────────────────────────────┐
│  Controller  │      │                    Policy                      │
│  Interface   │─────>│  normalize(numpy) → torch → inference → numpy  │
│  (robot I/O) │      └────────────────────────────────────────────────┘
└──────────────┘                          │
     ▲                                    │ numpy actions
     │ publish_action()                   ▼
     └────────────────────────────────────┘
                                          │
                                          ▼
                                   Episode Recorder
                                   (→ Ray Queue → Labeler → Replay Buffer)
```

All data entering the policy is **numpy arrays**. Torch tensors exist only inside the policy. The policy returns numpy actions.

## Subdirectories

| Directory | Purpose | README |
|-----------|---------|--------|
| [`auto/`](auto/) | Autonomous inference algorithms (RTC, Sequential) | [auto/README.md](auto/README.md) |
| [`human_in_the_loop/`](human_in_the_loop/) | Human-guided inference with teleoperation and intervention | [human_in_the_loop/README.md](human_in_the_loop/README.md) |
| [`policy/`](policy/) | Policy protocol, registry, loader, and implementations | [policy/README.md](policy/README.md) |
| [`nom_stats_manager/`](nom_stats_manager/) | Numpy-only data normalization using dataset statistics | [nom_stats_manager/README.md](nom_stats_manager/README.md) |
| [`inference_engine_utils/`](inference_engine_utils/) | Action inpainting and guided inference utilities | [inference_engine_utils/README.md](inference_engine_utils/README.md) |
| [`robot_io_interface/`](robot_io_interface/) | Hardware abstraction for reading state and publishing actions | [robot_io_interface/README.md](robot_io_interface/README.md) |
| [`episode_recorder/`](episode_recorder/) | Records observation-action pairs during episodes | [episode_recorder/README.md](episode_recorder/README.md) |
| [`runtime_settings_configs/`](runtime_settings_configs/) | Robot-specific runtime parameters and topic configs | [runtime_settings_configs/README.md](runtime_settings_configs/README.md) |

## `auto/` vs `human_in_the_loop/`

Both subdirectories implement the same core loop (read state → infer → publish actions) but differ in how actions are produced:

- **`auto/`** — The policy runs autonomously. The control loop reads robot state and publishes policy-predicted actions without human intervention. Used for production online RL.
- **`human_in_the_loop/`** — Adds a teleoperation layer. A human operator can intervene via a pedal switch, and an action multiplexer blends policy and teleoperator actions. Used for human-guided data collection and safety-critical scenarios.

Both share the same policy, normalization, and robot I/O components.

## Interface / Bridge Pattern

Most subdirectories follow an **interface → bridge** pattern for robot abstraction:

```
Interface (robot-agnostic API)
    └── robots/
        ├── igris_b/  ← bridge implementation
        └── igris_c/  ← bridge implementation
```

The interface class dispatches to the correct bridge based on the `robot` argument (e.g., `"igris_b"` or `"igris_c"`). To add a new robot, implement the bridge under a new `robots/<robot_name>/` directory and add the import branch in the interface class.

## Invariants

1. **Normalization is inside the policy.** The policy's inference methods receive a `DataNormalizationInterface` and call it internally.
2. **All data entering the policy is numpy.** No torch tensors cross the policy boundary inward.
3. **Policy output is numpy.** No torch tensors cross the policy boundary outward.
4. **Guided inference is inside the policy.** Action inpainting (blending old + new action chunks) happens in `policy.guided_inference()`.
5. **The normalization manager uses only numpy.** It has no torch dependency.
