# `features/residual_rl` — Onboarding Documentation

These docs explain everything the branch `features/residual_rl` adds on top of `main`. They are written for a **junior engineer who is fluent in Python and PyTorch but new to RL and new to this codebase**. They live alongside the broader [outer-repo docs](../README.md); read those first if you have never touched `online_rl` before.

> **Code is the source of truth.** Every claim below links into the working tree or the trainer submodule. If the code disagrees with these docs, the code wins — please open a PR.

---

## Table of contents

- [TL;DR](#tldr)
- [What changed vs main — at a glance](#what-changed-vs-main--at-a-glance)
- [Full diff inventory](#full-diff-inventory)
- [How to read these docs](#how-to-read-these-docs)
- [Document index](#document-index)
- [Branch metadata](#branch-metadata)
- [Caveats](#caveats)

---

## TL;DR

The branch adds **online residual reinforcement learning** to the existing `online_rl` stack. Concretely:

1. A new policy class, [`ResidualPolicy`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py), runs inside the **control loop** (40 Hz) of the [RTC inference algorithm](../02_architecture.md). It produces a small delta on top of the action chunk emitted by the base policy (currently DSRL-OpenPI). The combined action `base + residual` is what is sent to the robot.
2. A new replay-buffer actor, [`ResfitReplayBufferActor`](../../data_bridge/resfit_replay_buffer.py), records both the executed action and the **base action** at every step, then exposes LeRobot-style samples with action and reward *chunks* for off-line + on-policy training.
3. The [trainer submodule](https://github.com/KyunHwan/trainer) gains a new training recipe — [`resfit_trainer`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py) — that learns the residual via deterministic actor-critic with a Q-ensemble and an n-step TD critic. Only the residual actor's weights are pushed back to the env_actor; the base policy is frozen.
4. An optional **human-in-the-loop (HIL) path** lets a teleoperator transparently override the policy with a foot-pedal switch and a Manus glove + master-arm; transitions between policy and teleop are smoothly interpolated.
5. The `RTCActor` was already factored into a control loop and an inference loop; this branch lifts that split into the HIL package as well, deletes the dead `human_in_the_loop/io_interface` tree, and rewires the HIL packages to import from `env_actor.auto.*` and `env_actor.robot_io_interface.*` instead of maintaining duplicates.

There are **no published results** in this repository. The branch is the engineering scaffolding for residual RL on the igris_b hardware; treat any number you read as a hyper-parameter, not a benchmark.

---

## What changed vs `main` — at a glance

The branch touches **58 files in the parent repo** (`git diff --name-status main..features/residual_rl`). The trainer submodule pointer is the **same** on both branches (`3ca051a`), but the residual-RL training code lives at that SHA; see [Caveats](#caveats) and [04_trainer_submodule_changes.md](./04_trainer_submodule_changes.md).

The changes break down into seven groups. Each row points at the doc that explains it:

| # | Change group | Files | Where it is explained |
|---|---|---|---|
| **1** | **New residual policy package** (the inference-side residual `nn.Module`) | `env_actor/policy/policies/resfit_policy/{__init__.py, resfit_policy.py, resfit_policy.yaml, components/resfit_residual_actor.yaml}` | [01_concepts_residual_rl.md](./01_concepts_residual_rl.md) — what residual RL is and how it maps to this code <br/> [03_module_walkthrough.md § policies](./03_module_walkthrough.md#env_actorpolicypolicies) — file-by-file <br/> [05_config_and_hyperparameters.md § residual policy YAML](./05_config_and_hyperparameters.md#residual-policy-yaml) — the YAML fields |
| **2** | **New replay buffer** that stores both `action` and `base_policy_action` and emits chunked LeRobot-style samples | `data_bridge/resfit_replay_buffer.py` (new); `data_bridge/replay_buffer.py` (modified) | [03_module_walkthrough.md § data_bridge](./03_module_walkthrough.md#data_bridge) <br/> [04_trainer_submodule_changes.md § integration contract](./04_trainer_submodule_changes.md#integration-contract-with-the-parent-repo) <br/> [05_config_and_hyperparameters.md § replay-buffer args](./05_config_and_hyperparameters.md#replay-buffer-construction-args) |
| **3** | **Auto RTC actor threads residual through the control loop**: control loop now owns the residual policy, polls weights, applies `action = base + residual`; inference loop's weight reload is gated on `use_residual_rl=False`; shm bridge gains `init_action_chunk_obs_history` | `env_actor/auto/inference_algorithms/rtc/{rtc_actor.py, actors/control_loop.py, actors/inference_loop.py, data_manager/robots/igris_b/shm_manager_bridge.py, data_manager/shm_manager_interface.py}` | [02_architecture_changes.md § before/after diagrams](./02_architecture_changes.md#before--main) <br/> [03_module_walkthrough.md § auto rtc](./03_module_walkthrough.md#env_actorautoinference_algorithmsrtc) <br/> [06_data_flow_and_lifecycle.md](./06_data_flow_and_lifecycle.md) — the per-step and per-episode lifecycle |
| **4** | **Episode recorder writes `base_policy_action` and a task string**; new keys flow all the way to the trainer | `env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py` | [03_module_walkthrough.md § episode_recorder](./03_module_walkthrough.md#env_actorepisode_recorder) <br/> [06_data_flow_and_lifecycle.md § naming](./06_data_flow_and_lifecycle.md#where-data-is-shaped-and-where-it-is-named) |
| **5** | **HIL package rewrite**: new RTC actor and inference/control split for HIL; smooth interpolated POLICY↔TELEOP handoff via robot-specific `ActionInterpolator`; per-node executors for master arm and glove; tolerant Dynamixel sync-read; sequential actor cleanup; pedal switch semantics simplified | `env_actor/human_in_the_loop/{action_mux/**, inference_algorithms/**, intervention_methods/**, teleoperation/**}` (29 files, many deletions) | [02_architecture_changes.md § component table](./02_architecture_changes.md#what-changed-component-by-component) <br/> [03_module_walkthrough.md § action_mux, hil/rtc, hil/sequential, teleoperation, intervention_methods](./03_module_walkthrough.md#env_actorhuman_in_the_loopaction_mux) |
| **6** | **`io_interface/` tree deleted** — the HIL package no longer maintains its own duplicate of `controller_interface`, `controller_bridge`, `camera_utils`, `data_dict`, igris_c stubs; it now imports from the canonical `env_actor/robot_io_interface/` instead. The canonical interface gains a `ros_node` accessor and the bridge no longer calls `rclpy.shutdown()` itself. | All `env_actor/human_in_the_loop/io_interface/**` (deleted); `env_actor/robot_io_interface/{controller_interface.py, robots/igris_b/controller_bridge.py}` (modified) | [03_module_walkthrough.md § io_interface (deleted)](./03_module_walkthrough.md#env_actorhuman_in_the_loopio_interface-deleted) <br/> [03_module_walkthrough.md § robot_io_interface](./03_module_walkthrough.md#env_actorrobot_io_interface) |
| **7** | **Entry-point + cluster wiring**: `run_online_rl.py` adds `--use_residual_rl`, `--residual_policy_yaml`, `--use_human_intervention`; instantiates the new buffer; runs `run_training` synchronously; `start_ray.sh` raises symbolic resource quotas to `100`; three DSRL-OpenPI YAMLs are re-pointed to match the actual base-policy checkpoint. | `run_online_rl.py`, `start_ray.sh`, `env_actor/policy/policies/dsrl_openpi_policy/components/{noise_actor.yaml, noise_processor.yaml, openpi_model.yaml}` | [03_module_walkthrough.md § Root-level files](./03_module_walkthrough.md#root-level-files) <br/> [05_config_and_hyperparameters.md § entry-point CLI flags](./05_config_and_hyperparameters.md#entry-point-cli-flags) and [§ DSRL-OpenPI YAML deltas](./05_config_and_hyperparameters.md#dsrl-openpi-yaml-deltas) <br/> [07_running_training.md](./07_running_training.md) — how to invoke the new flags |

In addition, the **trainer submodule at `3ca051a`** carries the training-side scaffolding (`resfit_trainer`, `Critic_Trainer`, `Actor_Trainer`, `Residual_Actor`, `Q_Function`, the LeRobot dataloader, and `resfit.yaml`). All of that is documented at pinned SHAs in [04_trainer_submodule_changes.md](./04_trainer_submodule_changes.md). The submodule pointer was bumped to `3ca051a` on both `main` and `features/residual_rl` at the same time, so this is not a "submodule pointer change unique to the feature branch" — see [Caveats](#caveats) for the full explanation.

## Full diff inventory

Every file from `git diff --name-status main..features/residual_rl`, with the doc + section that covers it. Use this table when you want to look up a specific change rather than read top-down.

### Added (`A`) — 11 files

| File | Covering doc(s) |
|---|---|
| [data_bridge/resfit_replay_buffer.py](../../data_bridge/resfit_replay_buffer.py) | [03 § data_bridge](./03_module_walkthrough.md#data_bridge) + [main vs features/residual_rl side-by-side](./03_module_walkthrough.md#main-replay_bufferpy-vs-featuresresidual_rl-resfit_replay_bufferpy), [04 integration](./04_trainer_submodule_changes.md#integration-contract-with-the-parent-repo), [05 buffer args](./05_config_and_hyperparameters.md#replay-buffer-construction-args), [06 flow](./06_data_flow_and_lifecycle.md) |
| [env_actor/human_in_the_loop/action_mux/interp_utils/__init__.py](../../env_actor/human_in_the_loop/action_mux/interp_utils/__init__.py) | [03 § action_mux](./03_module_walkthrough.md#env_actorhuman_in_the_loopaction_mux) |
| [env_actor/human_in_the_loop/action_mux/interp_utils/interp_interface.py](../../env_actor/human_in_the_loop/action_mux/interp_utils/interp_interface.py) | [03 § action_mux — interp_interface.py](./03_module_walkthrough.md#interp_utilsinterp_interfacepy--new) |
| [env_actor/human_in_the_loop/action_mux/interp_utils/robots/igris_b/__init__.py](../../env_actor/human_in_the_loop/action_mux/interp_utils/robots/igris_b/__init__.py) | [03 § action_mux](./03_module_walkthrough.md#env_actorhuman_in_the_loopaction_mux) |
| [env_actor/human_in_the_loop/action_mux/interp_utils/robots/igris_b/igris_b_interpolator.py](../../env_actor/human_in_the_loop/action_mux/interp_utils/robots/igris_b/igris_b_interpolator.py) | [03 § action_mux — igris_b interpolator](./03_module_walkthrough.md#interp_utilsrobotsigris_bigris_b_interpolatorpy--new), [10 Q16](./10_faq_onboarding.md#q16-the-action-mux-uses-shortest-arc-interpolation-for-joints-why) |
| [env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py) | [03 § HIL rtc](./03_module_walkthrough.md#env_actorhuman_in_the_loopinference_algorithmsrtc) |
| [env_actor/human_in_the_loop/inference_algorithms/rtc/actors/inference_loop.py](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/inference_loop.py) | [03 § HIL rtc](./03_module_walkthrough.md#env_actorhuman_in_the_loopinference_algorithmsrtc) |
| [env_actor/human_in_the_loop/inference_algorithms/rtc/rtc_actor.py](../../env_actor/human_in_the_loop/inference_algorithms/rtc/rtc_actor.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § HIL rtc — rtc_actor.py](./03_module_walkthrough.md#rtc_actorpy--new) |
| [env_actor/policy/policies/resfit_policy/components/resfit_residual_actor.yaml](../../env_actor/policy/policies/resfit_policy/components/resfit_residual_actor.yaml) | [03 § policies — resfit_policy/](./03_module_walkthrough.md#env_actorpolicypolicies), [05 residual policy YAML](./05_config_and_hyperparameters.md#residual-policy-yaml) |
| [env_actor/policy/policies/resfit_policy/resfit_policy.py](../../env_actor/policy/policies/resfit_policy/resfit_policy.py) | [01 concept-to-code map](./01_concepts_residual_rl.md#concept-to-code-mapping), [03 § policies — resfit_policy/](./03_module_walkthrough.md#env_actorpolicypolicies), [10 Q3](./10_faq_onboarding.md#q3-where-is-the-residual-policy-actually-loaded) |
| [env_actor/policy/policies/resfit_policy/resfit_policy.yaml](../../env_actor/policy/policies/resfit_policy/resfit_policy.yaml) | [03 § policies — resfit_policy/](./03_module_walkthrough.md#env_actorpolicypolicies), [05 residual policy YAML](./05_config_and_hyperparameters.md#residual-policy-yaml) |

### Renamed (`R`) — 5 files

| Old path → new path | Covering doc(s) |
|---|---|
| `env_actor/human_in_the_loop/io_interface/__init__.py` → `env_actor/human_in_the_loop/action_mux/interp_utils/robots/__init__.py` | [03 § action_mux](./03_module_walkthrough.md#env_actorhuman_in_the_loopaction_mux), [03 § io_interface deletions](./03_module_walkthrough.md#env_actorhuman_in_the_loopio_interface-deleted) |
| `env_actor/human_in_the_loop/io_interface/robots/__init__.py` → `env_actor/human_in_the_loop/inference_algorithms/rtc/actors/__init__.py` | same |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/inference_engine_utils/max_deque.py` → `env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/utils/max_deque.py` | [03 § HIL rtc — Renames](./03_module_walkthrough.md#renames) |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/utils/utils.py` → `env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py` | [03 § HIL rtc — Renames](./03_module_walkthrough.md#renames) |
| `env_actor/human_in_the_loop/io_interface/robots/igris_b/__init__.py` → `env_actor/policy/policies/resfit_policy/__init__.py` | [03 § policies — resfit_policy/](./03_module_walkthrough.md#env_actorpolicypolicies), [03 § io_interface deletions](./03_module_walkthrough.md#env_actorhuman_in_the_loopio_interface-deleted) |

### Modified (`M`) — 27 files

| File | Covering doc(s) |
|---|---|
| [data_bridge/replay_buffer.py](../../data_bridge/replay_buffer.py) | [03 § data_bridge — replay_buffer.py](./03_module_walkthrough.md#data_bridgereplay_bufferpy--modified) |
| [env_actor/auto/inference_algorithms/rtc/actors/control_loop.py](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § auto rtc — control_loop.py](./03_module_walkthrough.md#actorscontrol_looppy--modified), [06 lifecycle](./06_data_flow_and_lifecycle.md) |
| [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § auto rtc — inference_loop.py](./03_module_walkthrough.md#actorsinference_looppy--modified) |
| [env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) | [03 § auto rtc — shm_manager_bridge.py](./03_module_walkthrough.md#data_managerrobotsigris_bshm_manager_bridgepy--modified) |
| [env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py](../../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py) | [03 § auto rtc — shm_manager_interface.py](./03_module_walkthrough.md#data_managershm_manager_interfacepy--modified) |
| [env_actor/auto/inference_algorithms/rtc/rtc_actor.py](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § auto rtc — rtc_actor.py](./03_module_walkthrough.md#rtc_actorpy--modified) |
| [env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py](../../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py) | [03 § episode_recorder](./03_module_walkthrough.md#env_actorepisode_recorder), [06 naming hops](./06_data_flow_and_lifecycle.md#where-data-is-shaped-and-where-it-is-named) |
| [env_actor/human_in_the_loop/action_mux/action_mux.py](../../env_actor/human_in_the_loop/action_mux/action_mux.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § action_mux — action_mux.py](./03_module_walkthrough.md#action_muxpy--modified), [10 Q16](./10_faq_onboarding.md#q16-the-action-mux-uses-shortest-arc-interpolation-for-joints-why) |
| [env_actor/human_in_the_loop/action_mux/intervention_switch.py](../../env_actor/human_in_the_loop/action_mux/intervention_switch.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § action_mux — intervention_switch.py](./03_module_walkthrough.md#intervention_switchpy--modified) |
| [env_actor/human_in_the_loop/action_mux/teleop_provider.py](../../env_actor/human_in_the_loop/action_mux/teleop_provider.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § action_mux — teleop_provider.py](./03_module_walkthrough.md#teleop_providerpy--modified) |
| [env_actor/human_in_the_loop/inference_algorithms/rtc/__init__.py](../../env_actor/human_in_the_loop/inference_algorithms/rtc/__init__.py) | [03 § HIL rtc — __init__.py](./03_module_walkthrough.md#__init__py--modified) |
| [env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) | [03 § HIL rtc — shm_manager_bridge.py](./03_module_walkthrough.md#data_managerrobotsigris_bshm_manager_bridgepy--modified-1) |
| [env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/shm_manager_interface.py](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/shm_manager_interface.py) | [03 § HIL rtc — package restructure](./03_module_walkthrough.md#env_actorhuman_in_the_loopinference_algorithmsrtc) |
| [env_actor/human_in_the_loop/inference_algorithms/sequential/data_manager/data_manager_interface.py](../../env_actor/human_in_the_loop/inference_algorithms/sequential/data_manager/data_manager_interface.py) | [03 § HIL sequential — data_manager_interface.py](./03_module_walkthrough.md#data_managerdata_manager_interfacepy--modified) |
| [env_actor/human_in_the_loop/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py](../../env_actor/human_in_the_loop/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py) | [03 § HIL sequential — igris_b bridge](./03_module_walkthrough.md#data_managerrobotsigris_bdata_manager_bridgepy--modified) |
| [env_actor/human_in_the_loop/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py](../../env_actor/human_in_the_loop/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py) | [03 § HIL sequential — igris_c bridge](./03_module_walkthrough.md#data_managerrobotsigris_cdata_manager_bridgepy--modified) |
| [env_actor/human_in_the_loop/inference_algorithms/sequential/sequential_actor.py](../../env_actor/human_in_the_loop/inference_algorithms/sequential/sequential_actor.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § HIL sequential — sequential_actor.py](./03_module_walkthrough.md#sequential_actorpy--modified) |
| [env_actor/human_in_the_loop/intervention_methods/pedal/publisher/pedal_publisher.py](../../env_actor/human_in_the_loop/intervention_methods/pedal/publisher/pedal_publisher.py) | [03 § pedal — publisher](./03_module_walkthrough.md#publisherpedal_publisherpy--modified) |
| [env_actor/human_in_the_loop/intervention_methods/pedal/subscriber/pedal_subscriber.py](../../env_actor/human_in_the_loop/intervention_methods/pedal/subscriber/pedal_subscriber.py) | [03 § pedal — subscriber](./03_module_walkthrough.md#subscriberpedal_subscriberpy--modified) |
| [env_actor/human_in_the_loop/teleoperation/robots/igris_b/arms_dynamixel.py](../../env_actor/human_in_the_loop/teleoperation/robots/igris_b/arms_dynamixel.py) | [02 component table](./02_architecture_changes.md#what-changed-component-by-component), [03 § teleoperation](./03_module_walkthrough.md#env_actorhuman_in_the_loopteleoperation) |
| [env_actor/policy/policies/dsrl_openpi_policy/components/noise_actor.yaml](../../env_actor/policy/policies/dsrl_openpi_policy/components/noise_actor.yaml) | [03 § policies — noise_actor](./03_module_walkthrough.md#dsrl_openpi_policycomponentsnoise_actoryaml--modified), [05 DSRL-OpenPI deltas](./05_config_and_hyperparameters.md#dsrl-openpi-yaml-deltas) |
| [env_actor/policy/policies/dsrl_openpi_policy/components/noise_processor.yaml](../../env_actor/policy/policies/dsrl_openpi_policy/components/noise_processor.yaml) | [03 § policies — noise_processor](./03_module_walkthrough.md#dsrl_openpi_policycomponentsnoise_processoryaml--modified), [05 DSRL-OpenPI deltas](./05_config_and_hyperparameters.md#dsrl-openpi-yaml-deltas) |
| [env_actor/policy/policies/dsrl_openpi_policy/components/openpi_model.yaml](../../env_actor/policy/policies/dsrl_openpi_policy/components/openpi_model.yaml) | [03 § policies — openpi_model](./03_module_walkthrough.md#dsrl_openpi_policycomponentsopenpi_modelyaml--modified), [05 DSRL-OpenPI deltas](./05_config_and_hyperparameters.md#dsrl-openpi-yaml-deltas) |
| [env_actor/robot_io_interface/controller_interface.py](../../env_actor/robot_io_interface/controller_interface.py) | [03 § robot_io_interface — controller_interface](./03_module_walkthrough.md#controller_interfacepy--modified) |
| [env_actor/robot_io_interface/robots/igris_b/controller_bridge.py](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) | [03 § robot_io_interface — controller_bridge](./03_module_walkthrough.md#robotsigris_bcontroller_bridgepy--modified) |
| [run_online_rl.py](../../run_online_rl.py) | [03 § Root-level — run_online_rl.py](./03_module_walkthrough.md#run_online_rlpy--modified), [05 CLI flags](./05_config_and_hyperparameters.md#entry-point-cli-flags), [07 quickstart](./07_running_training.md), [10 Q11](./10_faq_onboarding.md#q11-why-is-run_training-not-a-ray-task-anymore) |
| [start_ray.sh](../../start_ray.sh) | [02 Ray resource labels](./02_architecture_changes.md#ray-resource-labels), [03 § Root-level — start_ray.sh](./03_module_walkthrough.md#start_raysh--modified), [07 Start Ray](./07_running_training.md#start-ray) |

### Deleted (`D`) — 15 files

| File | Covering doc(s) |
|---|---|
| `env_actor/human_in_the_loop/inference_algorithms/rtc/control_actor.py` | [03 § HIL rtc — Deletions inside rtc/](./03_module_walkthrough.md#deletions-inside-rtc) |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/inference_actor.py` | same |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/inference_engine_utils/__init__.py` | same |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/inference_engine_utils/action_inpainting.py` | same |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/data_normalization_interface.py` | same |
| `env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/data_normalization_manager.py` | same |
| `env_actor/human_in_the_loop/io_interface/controller_interface.py` | [03 § io_interface/ (deleted)](./03_module_walkthrough.md#env_actorhuman_in_the_loopio_interface-deleted) |
| `env_actor/human_in_the_loop/io_interface/robots/igris_b/controller_bridge.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_b/utils/__init__.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_b/utils/camera_utils.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_b/utils/data_dict.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_c/__init__.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_c/controller_bridge.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_c/igris_c.py` | same |
| `env_actor/human_in_the_loop/io_interface/robots/igris_c/utils/__init__.py` | same |

Total: **11 added + 5 renamed + 27 modified + 15 deleted = 58 files**, matching `git diff --name-status main..features/residual_rl` exactly.

---

## How to read these docs

If you have **one hour**, read in this order. Times are reading time only:

| # | File | Reading time | When you finish, you can answer |
|---|---|---|---|
| 1 | [00_overview.md](./00_overview.md) | 5 min | *Why does this branch exist? What is in scope?* |
| 2 | [01_concepts_residual_rl.md](./01_concepts_residual_rl.md) | 10 min | *What is residual RL? Where do the terms map into code?* |
| 3 | [02_architecture_changes.md](./02_architecture_changes.md) | 8 min | *Which Ray actors and processes changed? How do they fit together now?* |
| 4 | [06_data_flow_and_lifecycle.md](./06_data_flow_and_lifecycle.md) | 10 min | *What happens from a sensor frame to a weight update?* |
| 5 | [07_running_training.md](./07_running_training.md) | 7 min | *How do I actually launch a residual-RL training run on my desk?* |

Once you have the shape of the system, dip into the reference docs as you need them:

| # | File | Use when... |
|---|---|---|
| 6 | [03_module_walkthrough.md](./03_module_walkthrough.md) | You need to find or change the implementation of any specific module touched by this branch. |
| 7 | [04_trainer_submodule_changes.md](./04_trainer_submodule_changes.md) | You are debugging the training-side loss, the Q-function, or the actor optimizer. |
| 8 | [05_config_and_hyperparameters.md](./05_config_and_hyperparameters.md) | You want to know what a knob does before turning it. |
| 9 | [08_debugging_and_observability.md](./08_debugging_and_observability.md) | Something is broken: logs, metrics, common failure modes, how to bisect. |
| 10 | [09_glossary.md](./09_glossary.md) | You hit an unfamiliar term. |
| 11 | [10_faq_onboarding.md](./10_faq_onboarding.md) | You have a "how do I…?" question after Day 1. |

---

## Document index

Each entry below names the **questions** that doc answers, not just its title — use it as a router into the rest of the set.

### [00_overview.md](./00_overview.md)
*Why was this branch created? What is and isn't in scope? Where does the residual policy plug into the existing system?* Also lists the things this branch deliberately does **not** change so you know what to leave alone.

### [01_concepts_residual_rl.md](./01_concepts_residual_rl.md)
*What is RL — in one screen? What does "residual policy" mean? Why is it appealing for robotics? Which textbook term maps to which variable name and file in this repo?* Includes a worked toy example, and a "common misconceptions" list aimed at engineers coming from supervised learning.

### [02_architecture_changes.md](./02_architecture_changes.md)
*What is the Ray-actor graph on `main`, and what is it now? Which processes spawn which? Where does the residual policy live, and where does the trainer live?* Two side-by-side Mermaid diagrams (before / after) and a component-by-component breakdown.

### [03_module_walkthrough.md](./03_module_walkthrough.md)
*For each top-level directory the branch touches, what files were added, deleted, or modified, and what does each new symbol do?* Organized as one section per directory (`data_bridge/`, `env_actor/auto/…/rtc/`, `env_actor/human_in_the_loop/`, `env_actor/policy/`, `env_actor/robot_io_interface/`, the entry-point scripts).

### [04_trainer_submodule_changes.md](./04_trainer_submodule_changes.md)
*What does the trainer submodule contribute to residual RL? What does `resfit_trainer` actually optimize? How is the Q-function trained? Which weights flow back to the env_actor?* Submodule pointer information is here — read it before assuming the submodule was bumped on this branch (it wasn't, see [Caveats](#caveats)).

### [05_config_and_hyperparameters.md](./05_config_and_hyperparameters.md)
*What new YAMLs / CLI flags exist, and what do their fields mean? Which defaults are "leave alone," which are "tune to your task"?* Single reference table for every new key, plus a section on precedence (CLI > YAML > code default).

### [06_data_flow_and_lifecycle.md](./06_data_flow_and_lifecycle.md)
*From a sensor frame to a weight update, what happens step by step?* Mermaid sequence diagram covers the rollout → replay-buffer → trainer → weight-push → control-loop reload cycle, including how the base-policy chunk and the residual delta are recorded and re-used during training.

### [07_running_training.md](./07_running_training.md)
*Prereqs, install, smoke test, full training command, where outputs land, what "healthy" looks like.* Everything a junior needs for their first day, with copy-pasteable commands and expected log snippets.

### [08_debugging_and_observability.md](./08_debugging_and_observability.md)
*Which metrics matter? Which log lines are signal versus noise? What does the system look like when each common failure happens (NaN losses, replay buffer stuck at size 0, actor stalls, shared-memory cleanup failures, base-policy checkpoint missing)?* Includes recipes for running one component in isolation.

### [09_glossary.md](./09_glossary.md)
*One-line definitions for every acronym and jargon term used in these docs.* Pruned to terms actually present in the codebase — no decorative entries.

### [10_faq_onboarding.md](./10_faq_onboarding.md)
*Concrete answers to the questions a junior will hit on Days 2–5*: how to add a reward term, how to swap the residual network, how to resume from a checkpoint, why there used to be two run scripts, what the `online_update` config flag actually toggles.

---

## Branch metadata

| Field | Value |
|---|---|
| Parent-repo working branch | `features/residual_rl` |
| Parent-repo HEAD (at doc capture) | `414291b209b85d9b6794e259822f5ae0434e71c2` |
| Baseline branch | `main` |
| Baseline HEAD (at doc capture) | `85fb2e9b44ac08f039f10ca7586e5fd36b36f105` |
| Merge base | `a7adf1d6ee7cae095bd9b5612f7a480befbdb5a9` |
| Commits unique to feature branch | 69 |
| `trainer` submodule SHA on both branches | `3ca051a256c9068f77b556df98f538d9a6185ccf` (identical — see [Caveats](#caveats)) |
| `trainer` submodule remote | https://github.com/KyunHwan/trainer |
| `data_labeler/auto/models/robometer` SHA on both branches | `a3d08d1f9821eb57154b3146477f2bd405cea283` (unchanged) |
| GitHub compare URL | https://github.com/KyunHwan/online_rl/compare/main...features/residual_rl |
| Files changed (parent repo only) | 58 (`git diff --stat main..features/residual_rl`) |
| Diff capture date | 2026-05-13 |

To reproduce the diff used while writing these docs:

```bash
git fetch origin
git log --oneline main..features/residual_rl
git diff --stat main..features/residual_rl
git diff --name-status main..features/residual_rl
```

---

## Caveats

These are facts that surprised the doc author while reading the live repo. They are deliberately surfaced here so future readers do not waste time:

- **The `trainer` submodule pointer is identical on both branches.** Both `main` and `features/residual_rl` currently point at `3ca051a`. The §0 prompt that produced these docs assumed a submodule pointer bump existed on this branch; it does not. The residual-RL training code is nevertheless real and lives inside the submodule at that SHA — it landed on `main` at the same time, via `main`'s commit `36727f1 updated trainer version`. When you read [04_trainer_submodule_changes.md](./04_trainer_submodule_changes.md), interpret the "changes" as *the trainer-side scaffolding required for residual RL, present on both branches at this SHA*, not as a diff that lives only on `features/residual_rl`.
- **The branch is behind `main` on a handful of cleanups** that were already merged forward (e.g., `main`'s `01e9219 removed redundant run_online_rl_openpi.py` and `a7adf1d added numpy float32 check for shm_manager_bridge`). The feature branch removed `run_online_rl_openpi.py` itself in `b9369a4`, so the file is gone on both sides; the bookkeeping just happened along different paths.
- **There is no entry point called `run_online_rl_openpi.py`.** The §0 prompt mentioned it; the live repo no longer contains it (it was deleted on both branches). The only training entry point is [`run_online_rl.py`](../../run_online_rl.py).
- **There is a second submodule** the §0 prompt did not mention: `data_labeler/auto/models/robometer`. Its pointer is unchanged on this branch, so the new docs ignore it. It exists for the auto reward labeler used during training.
- **The `trainer/` subdir is the submodule's working copy on disk.** Inside it you will see `trainer/` again (the inner Python package). Both are real and the doubled name is intentional — top-level `trainer/` is the *repo*, inner `trainer/trainer/` is the *package*.
- **No published result.** Numbers in YAMLs (learning rates, horizons, episode lengths, the `np.random.uniform(-0.08, 0.08)` initial-noise residual, etc.) are *settings used during development*, not validated optima.

If you spot another mismatch, fix the doc and note it here — `README.md` is the index, not the place to bury bug reports.
