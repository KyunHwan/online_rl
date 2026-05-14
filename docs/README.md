# Documentation index

This is the hub for the `online_rl` outer-repository documentation. The numbered docs (`01`..`10`) are the primary reading order; per-folder `README.md` files are reference material you jump to when you are already inside a directory.

If you are new to this codebase, read [01_getting_started.md](01_getting_started.md) and [02_architecture.md](02_architecture.md) in that order, then dip into whatever you need.

## Getting started

| If you are here to... | Read |
|---|---|
| Install, smoke-test, and run your first job | [01_getting_started.md](01_getting_started.md) |
| Understand the Ray actor graph end-to-end | [02_architecture.md](02_architecture.md) |
| Set up the Tailscale-based 3-machine cluster | [03_distributed_setup.md](03_distributed_setup.md) |

## Reference

| Document | What's in it |
|---|---|
| [04_configuration.md](04_configuration.md) | The three config layers: train YAML, policy YAML, runtime JSON |
| [05_policy_protocol.md](05_policy_protocol.md) | The `Policy` Protocol a new policy must satisfy |
| [06_data_flow.md](06_data_flow.md) | One episode's data from sensor to weight update, hop by hop |
| [08_invariants.md](08_invariants.md) | Architectural rules the extending engineer must respect |
| [10_glossary.md](10_glossary.md) | Terminology cheatsheet (Ray, TensorDict, RTC, DSRL, OpenPI, ...) |

## Operations and extension

| Document | What's in it |
|---|---|
| [07_extending.md](07_extending.md) | Recipes: new policy, new robot, new inference algorithm, new reward labeler |
| [09_troubleshooting.md](09_troubleshooting.md) | Symptom-first decision tree of failure modes |

## Per-folder reference

Every directory inside the outer repo has a local README. Open these when you are already in the folder:

- Top-level: [`env_actor/`](../env_actor/README.md), [`data_bridge/`](../data_bridge/README.md), [`data_labeler/`](../data_labeler/README.md)
- Inference: [`env_actor/auto/`](../env_actor/auto/README.md), [`env_actor/auto/inference_algorithms/`](../env_actor/auto/inference_algorithms/README.md)
- Policy: [`env_actor/policy/`](../env_actor/policy/README.md), [`env_actor/policy/registry/`](../env_actor/policy/registry/README.md)
- Robot I/O: [`env_actor/robot_io_interface/`](../env_actor/robot_io_interface/README.md), [`env_actor/episode_recorder/`](../env_actor/episode_recorder/README.md), [`env_actor/nom_stats_manager/`](../env_actor/nom_stats_manager/README.md), [`env_actor/inference_engine_utils/`](../env_actor/inference_engine_utils/README.md)
- Config: [`env_actor/runtime_settings_configs/`](../env_actor/runtime_settings_configs/README.md)
- Scaffolding (not currently wired in): [`env_actor/human_in_the_loop/`](../env_actor/human_in_the_loop/README.md)

## Branch-specific docs

These docs apply only when you are on the named branch. If you are on `main`, skip them.

| Branch | Doc set | What's in it |
|---|---|---|
| `features/residual_rl` | [residual_rl/README.md](residual_rl/README.md) | Onboarding for the online residual-RL feature branch: what changed vs `main`, the new residual policy and replay buffer, trainer-submodule scaffolding, and how to run a residual-RL training job. |

## The training half

The trainer is a [git submodule](10_glossary.md#git-submodule) with its own complete doc hierarchy. The outer docs do not duplicate it. For training-specific topics, go straight to [trainer/docs/README.md](../trainer/docs/README.md). The cross-references most relevant to the outer repo are:

- [trainer/docs/03_ray_online_training.md](../trainer/docs/03_ray_online_training.md) — the `replay_buffer` and `policy_state_manager` actor contracts from the trainer's side.
- [trainer/docs/04_concepts.md](../trainer/docs/04_concepts.md) — the Registry / Factory mental model that the env_actor's policy loader also uses.
- [trainer/docs/05_configuration.md](../trainer/docs/05_configuration.md) — the train-YAML schema consumed by `train_func()` in [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py).

## How the docs are organized

Code is the source of truth. Every claim in these docs cites the file (and where useful, the line range) that implements it. If a doc and the code disagree, the code wins — please open a PR fixing the doc.

Anything a junior reader would not already know is defined in [10_glossary.md](10_glossary.md). If you find a term that is not defined there, that is a gap worth filing.
