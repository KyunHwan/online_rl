# online_rl

`online_rl` is a distributed pipeline that trains a robot manipulation policy from data the same robot is producing right now. A policy runs on the robot, episodes flow into a [reward labeler](docs/10_glossary.md#reward-labeler) and then a replay buffer, a trainer samples from the buffer, and updated weights are pushed back to the running policy — all on a [Ray](docs/10_glossary.md#ray) cluster of three machines. The system exists because supervised pre-training alone does not recover from the distribution shift that real hardware introduces.

## Architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│                       Ray Cluster (3 machines)                        │
│                                                                       │
│   inference_pc              labeling_pc (HEAD)        training_pc     │
│                                                                       │
│   ┌───────────────┐         ┌─────────────────┐     ┌──────────────┐  │
│   │   EnvActor    │──ep──▶ │ AutoRewardLabeler│────▶│ ReplayBuffer │  │
│   │ RTC / Seq.    │ queue   │  (Robometer VLM) │     │ (memmap disk)│  │
│   └───────────────┘         └─────────────────┘     └──────┬───────┘  │
│         ▲                                                  │ sample() │
│         │ weights                                   ┌──────▼───────┐  │
│   ┌─────┴───────┐                                   │   Trainer    │  │
│   │   State     │◀──────────────────────────────────│ (Ray DDP x4) │  │
│   │  Manager    │            weight ref             └──────────────┘  │
│   └─────────────┘                                                     │
└───────────────────────────────────────────────────────────────────────┘
```

Each box is a [Ray actor](docs/10_glossary.md#ray-actor) pinned to a machine by [custom resource](docs/10_glossary.md#custom-resource) label. The `EnvActor` runs on the robot's inference machine, the trainer and `ReplayBuffer` / `StateManager` actors run on the training machine, and the auto-labeler runs on the labeling machine (which also serves as the cluster head — see [docs/03_distributed_setup.md](docs/03_distributed_setup.md)).

## Where to go next

| I want to... | Read |
|---|---|
| Get running in 30 minutes | [docs/01_getting_started.md](docs/01_getting_started.md) |
| Understand the architecture | [docs/02_architecture.md](docs/02_architecture.md) |
| Add a new policy / robot / inference algorithm | [docs/07_extending.md](docs/07_extending.md) |
| Debug why my run is broken | [docs/09_troubleshooting.md](docs/09_troubleshooting.md) |
| Look up a term I don't know | [docs/10_glossary.md](docs/10_glossary.md) |
| Browse all the docs | [docs/README.md](docs/README.md) |
| Understand how the training half works | [trainer/docs/README.md](trainer/docs/README.md) |
| Run training standalone, without the env actor | [trainer/docs/01_getting_started.md](trainer/docs/01_getting_started.md) |

## Repository layout

| Path | Purpose | README |
|---|---|---|
| [run_online_rl.py](run_online_rl.py) | The only entrypoint. Spawns every actor. | — |
| [start_ray.sh](start_ray.sh) | Starts Ray on each machine via hostname-based routing. | — |
| [env_setup.sh](env_setup.sh) / [uv_setup.sh](uv_setup.sh) | One-shot dependency install. | — |
| [openpi_transformer_lib_patch.sh](openpi_transformer_lib_patch.sh) | Patches HuggingFace `transformers` to add OpenPI-specific layers. | — |
| [env_actor/](env_actor/) | Policy inference, robot I/O, normalization, episode recording. | [env_actor/README.md](env_actor/README.md) |
| [data_bridge/](data_bridge/) | `ReplayBufferActor` (memmap disk) and `StateManagerActor` (weight transport). | [data_bridge/README.md](data_bridge/README.md) |
| [data_labeler/](data_labeler/) | Reward annotation: VLM-based auto labeler or PySide6 GUI manual labeler. | [data_labeler/README.md](data_labeler/README.md) |
| [trainer/](trainer/) | Training library — git submodule. Owns the train loop, registries, configs. | [trainer/docs/README.md](trainer/docs/README.md) |
| [docs/](docs/) | Cross-cutting docs (this README is just the front door). | [docs/README.md](docs/README.md) |

## Quickstart

Three commands prove your environment works. The full walkthrough is in [docs/01_getting_started.md](docs/01_getting_started.md).

```bash
# 1. Clone with submodules.
git clone --recurse-submodules <repo-url> online_rl && cd online_rl

# 2. Install dependencies (requires sudo for apt packages).
bash uv_setup.sh && source $HOME/.local/bin/env && source .venv/bin/activate
bash env_setup.sh
bash openpi_transformer_lib_patch.sh

# 3. Smoke test — confirms imports without starting the cluster.
python -c "import ray, torch, tensordict, torchrl; from trainer.trainer.online_trainer import train_func; print('ok')"
```

After the smoke test, you start Ray on each machine and launch a run — both steps are covered in [docs/01_getting_started.md](docs/01_getting_started.md).

## Architectural rules

Extending this codebase requires respecting a handful of [invariants](docs/08_invariants.md) — most importantly that the [numpy/torch boundary lives at the policy interface](docs/08_invariants.md#all-data-crossing-the-policy-boundary-is-numpy) and that [normalization happens inside the policy](docs/08_invariants.md#normalization-is-inside-the-policy). Read that doc before adding a new policy or robot.

## Known doc/code drift

Three deployment-specific values in this repo are hard-coded to the original developer's machine. A new operator must edit them before the first run:

- **Tailscale head IP** in [start_ray.sh](start_ray.sh) — currently `100.118.28.46`.
- **Hostnames** in [start_ray.sh](start_ray.sh)'s `case "$HOSTNAME"` — `robros-MS-7E59`, `robros-ai1`, `robros-5090`.
- **`norm_stats_file_path`** in [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json](env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) — points at `/home/robros/Projects/inference_engine/...` on the original developer's box.

See [docs/01_getting_started.md](docs/01_getting_started.md) and [docs/03_distributed_setup.md](docs/03_distributed_setup.md) for the edits needed.

`--robot igris_c` is currently a stub: the runtime config files do not exist. Only `igris_b` works today. See [docs/09_troubleshooting.md](docs/09_troubleshooting.md).
