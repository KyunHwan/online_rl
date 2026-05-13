# env_actor/human_in_the_loop/inference_algorithms

> **Status: not wired in.** None of the files in this subtree are imported by [run_online_rl.py](../../../run_online_rl.py). The directory is scaffolding for a future HIL-aware entrypoint that does not exist today. See [../README.md](../README.md#status) for the parent-level status callout. The `--human_reward_labeler` flag does **not** activate this code — it only switches the reward labeler.

This directory mirrors [../../auto/inference_algorithms/](../../auto/inference_algorithms/), with the same RTC vs Sequential split, but the control side is extended with a teleoperation provider and an action multiplexer so a human can intervene during a policy run. Read this README only if you intend to finish the wiring and add an HIL entrypoint.

## Table of contents

- [What is here, what is not](#what-is-here-what-is-not)
- [RTC (HIL variant)](#rtc-hil-variant)
- [Sequential (HIL variant)](#sequential-hil-variant)
- [What's missing to activate this code](#whats-missing-to-activate-this-code)

## What is here, what is not

```text
inference_algorithms/
├── rtc/
│   ├── control_actor.py                  # paired with inference_actor.py
│   ├── inference_actor.py
│   ├── data_manager/
│   │   ├── data_normalization_interface.py   # NOTE: NOT the same as env_actor/nom_stats_manager/
│   │   ├── shm_manager_interface.py
│   │   └── robots/igris_b/
│   │       ├── data_normalization_manager.py
│   │       └── shm_manager_bridge.py
│   └── inference_engine_utils/
│       ├── action_inpainting.py
│       └── max_deque.py
└── sequential/
    ├── sequential_actor.py
    └── data_manager/
        ├── data_manager_interface.py
        └── robots/{igris_b,igris_c}/
            └── data_manager_bridge.py
```

This is the *scaffold*. It has not been kept in sync with the live `env_actor/auto/inference_algorithms/` tree — some files here duplicate functionality that the live tree later consolidated (for example, `data_normalization_interface.py` is a local copy of what [`env_actor/nom_stats_manager/data_normalization_interface.py`](../../nom_stats_manager/data_normalization_interface.py) became in the auto path). Treat anything here as a starting point for HIL work, not as a reference.

## RTC (HIL variant)

The intended design (not running today): a parent actor would spawn two child processes — one for inference, one for control + HIL action mux — communicating through shared memory, just like the auto-mode RTC. The control side would additionally consult:

- A teleoperation provider (e.g. [`../teleoperation/robots/igris_b/arms_dynamixel.py`](../teleoperation/robots/igris_b/arms_dynamixel.py) for the input-side arms, [`../teleoperation/robots/igris_b/hands_manus.py`](../teleoperation/robots/igris_b/hands_manus.py) for the input-side hands).
- An intervention switch (e.g. [`../intervention_methods/pedal/subscriber/pedal_subscriber.py`](../intervention_methods/pedal/subscriber/pedal_subscriber.py)).
- An action multiplexer ([`../action_mux/action_mux.py`](../action_mux/action_mux.py)) that decides whether to publish the policy's action or the teleoperator's based on the pedal state.

The `control_mode` field on each timestep would then mark autonomous vs HIL segments, and `EpisodeRecorderBridge._split_by_control_mode_as_episodes` would split runs at boundaries so the replay buffer's HIL-routing logic (`use_hil_buffer=True`) routes the two kinds of data into separate buffers.

## Sequential (HIL variant)

A single-process synchronous loop with the same HIL hooks. The action mux runs inline in the control loop, blending policy and teleop. Same `control_mode` tagging in the episode recorder.

## What's missing to activate this code

A new entrypoint (e.g. `run_online_rl_hil.py`, not present in the repo) would need to:

1. Import the HIL RTC or Sequential actor from this directory instead of `env_actor/auto/inference_algorithms/...`.
2. Set up the action mux, teleop providers, and intervention switch.
3. Pass them to the HIL actor as constructor arguments.
4. Optionally set `use_hil_buffer=True` on `ReplayBufferActor` to keep HIL and autonomous segments separate.

Until that entrypoint exists, nothing in this subtree is loaded.

For the live (auto) version of these algorithms, read [../../auto/inference_algorithms/README.md](../../auto/inference_algorithms/README.md).
