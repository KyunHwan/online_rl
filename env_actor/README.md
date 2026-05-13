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

- **`auto/`** is what [run_online_rl.py](../run_online_rl.py) actually drives. The policy runs autonomously: the control loop reads robot state and publishes policy-predicted actions without human intervention.
- **`human_in_the_loop/`** is scaffolding for a future HIL entrypoint that does not exist in the repo today. None of its files are imported by `run_online_rl.py`. Treat it as a parallel template, not as a live subsystem. See [human_in_the_loop/README.md](human_in_the_loop/README.md) for the Status callout and what is needed to activate it. Note that the `--human_reward_labeler` flag only switches the reward labeler — it does **not** activate any HIL inference code.

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

The architectural rules — normalization is inside the policy, numpy at the policy boundary, etc. — are documented with the *why* and the *what breaks* in [../docs/08_invariants.md](../docs/08_invariants.md). Read that before extending anything here.
