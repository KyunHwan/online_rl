# 01 — Getting started

This page walks a junior engineer from a clean Ubuntu box to a running online RL job on the supported robot (`igris_b`). It assumes you have not seen this codebase before. Read straight through.

> Heads-up — three values in this repo are hard-coded to the original developer's machine. You will edit them in steps 6 and 7. They are flagged inline.

## Table of contents

- [Prerequisites](#prerequisites)
- [1. Clone the repo with submodules](#1-clone-the-repo-with-submodules)
- [2. Install uv and create the virtualenv](#2-install-uv-and-create-the-virtualenv)
- [3. Install Python and system dependencies](#3-install-python-and-system-dependencies)
- [4. Patch the transformers library for OpenPI](#4-patch-the-transformers-library-for-openpi)
- [5. Smoke test the install](#5-smoke-test-the-install)
- [6. Edit the deployment-specific paths and hostnames](#6-edit-the-deployment-specific-paths-and-hostnames)
- [7. Edit the runtime JSON for your machine](#7-edit-the-runtime-json-for-your-machine)
- [8. Start the Ray cluster](#8-start-the-ray-cluster)
- [9. Launch a run](#9-launch-a-run)
- [10. Expected console output](#10-expected-console-output)
- [First-run gotchas](#first-run-gotchas)

## Prerequisites

| Requirement | Why |
|---|---|
| Ubuntu Linux (tested on Ubuntu 24.04) | `apt-get` packages are Debian/Ubuntu specific |
| Python 3.12.3 | [uv_setup.sh](../uv_setup.sh) pins `--python 3.12.3` |
| Sudo access | [env_setup.sh](../env_setup.sh) calls `sudo apt-get install` for `ffmpeg` and `libav*-dev` |
| CUDA-capable GPU on the inference and training machines | The inference actor and the trainer both run on GPU |
| [Tailscale](10_glossary.md#tailscale) on every machine | The default `start_ray.sh` joins nodes by Tailscale IP |
| 3 machines for a real deployment, OR 1 machine for development | The pipeline is multi-machine by design; single-machine works if all resource labels live on one host |

If you cannot run `sudo apt-get install` on your machine (e.g. managed infrastructure), ask an admin to pre-install: `ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev`. Comment out the corresponding line in [env_setup.sh](../env_setup.sh) before you run it.

## 1. Clone the repo with submodules

The outer repo has two [git submodules](10_glossary.md#git-submodule) — the trainer and Robometer. Both are required.

```bash
git clone --recurse-submodules <repo-url> online_rl
cd online_rl
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

You can confirm by listing `.gitmodules` — it points at `trainer/` and `data_labeler/auto/models/robometer/`.

## 2. Install uv and create the virtualenv

[uv_setup.sh](../uv_setup.sh) installs the [uv](https://docs.astral.sh/uv/) package manager and creates a `.venv` at the repo root pinned to Python 3.12.3:

```bash
bash uv_setup.sh
source $HOME/.local/bin/env
source .venv/bin/activate
```

From now on every `python` / `pip` invocation should go through this venv. Verify with `which python` — it should point at `<repo>/.venv/bin/python`.

## 3. Install Python and system dependencies

[env_setup.sh](../env_setup.sh) installs everything: PyTorch with CUDA 13.0 wheels, Ray, TensorDict, torchrl, the OpenPI-related JAX stack, LeRobot, the GUI labeler dependencies, and the Robometer submodule in editable mode.

```bash
bash env_setup.sh
```

What this does, in order:

1. `uv pip install torch==2.9.0 torchvision==0.24.0` from PyTorch's CUDA 13.0 wheels index.
2. `uv pip install flow_matching schedulefree geomloss einops wandb timm tensordict ray[default] pyvers cloudpickle torchrl`.
3. The OpenPI stack: `uv pip install transformers==4.53.2 pytest flax augmax beartype jaxtyping==0.2.34 sentencepiece chex tqdm-loggable numpydantic`.
4. The dataset stack: `uv pip install --no-deps lerobot datasets accelerate`.
5. `sudo apt-get update && sudo apt-get install ffmpeg libav*-dev libsw*-dev` — **this is the sudo step**.
6. `uv pip install torchcodec==0.9.1 av tyro ml_collections gcsfs`.
7. `uv pip install PySide6` for the manual reward labeler GUI.
8. `uv pip install -e ./data_labeler/auto/models/robometer` — installs the Robometer submodule as an editable package. The auto labeler also adds this path to `sys.path` at runtime in [data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py); both paths exist so the actor works whether or not the editable install survived a venv rebuild.
9. `uv pip install -e ./trainer/.../depth_anything_3` — the Depth Anything v3 vision backbone, also installed editably.

The install takes 5–15 minutes depending on network speed and CUDA wheel download time.

## 4. Patch the transformers library for OpenPI

OpenPI's PyTorch port replaces a few files inside the installed `transformers` package. The patch script copies them into `.venv/lib/python3.12/site-packages/transformers/`:

```bash
bash openpi_transformer_lib_patch.sh
```

This is required if you intend to load either of the OpenPI-based policies (`openpi_policy` or `dsrl_openpi_policy`). It assumes `transformers==4.53.2` was just installed by `env_setup.sh`. If you upgrade `transformers`, rerun the patch.

The script is short — read [openpi_transformer_lib_patch.sh](../openpi_transformer_lib_patch.sh) to see what it actually does.

## 5. Smoke test the install

Before starting the cluster, prove that imports work end-to-end:

```bash
python -c "import ray, torch, tensordict, torchrl; \
           from trainer.trainer.online_trainer import train_func; \
           from env_actor.policy.utils.loader import build_policy; \
           print('imports ok')"
```

If you see `imports ok`, the install is good. If you see `ModuleNotFoundError: trainer`, you are missing the submodule (`git submodule update --init --recursive`). If you see `transformers` errors, the patch did not run.

## 6. Edit the deployment-specific paths and hostnames

Open [start_ray.sh](../start_ray.sh). Three values are deployment-specific and must match your environment:

```bash
HEAD_IP="100.118.28.46"     # ← your head node's Tailscale IPv4
case "$HOSTNAME" in
  robros-MS-7E59)           # ← your labeling/head machine's hostname
  robros-ai1)               # ← your training machine's hostname
  robros-5090)              # ← your inference machine's hostname
```

The head machine is the one that runs `ray start --head`. In the script as committed, that is `robros-MS-7E59`, which is also the labeling machine. The other two machines join the head with `ray start --address=${HEAD_IP}:6379`.

The Tailscale IP is fetched dynamically with `tailscale ip -4`, but the **head** address `HEAD_IP` is the value workers will connect to — that one must be your head node's actual Tailscale address. Run `tailscale ip -4` on your head machine to get it.

If you are running on a single machine for development, the simplest workaround is to set all three hostnames to your machine's hostname (which collapses all three cases into one `ray start --head`) and skip the cross-machine networking entirely. The commented-out LAN block at the bottom of [start_ray.sh](../start_ray.sh) shows the same shape with `--address` rather than Tailscale.

See [03_distributed_setup.md](03_distributed_setup.md) for the full topology explanation.

## 7. Edit the runtime JSON for your machine

Open [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json). The last entry is `norm_stats_file_path`:

```json
"norm_stats_file_path": "/home/robros/Projects/inference_engine/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dataset_stats.pkl"
```

This absolute path points to the original developer's machine. **The file does not exist on yours.** You must:

1. Locate (or produce) the [normalization-statistics](10_glossary.md#normalization-stats) pickle file. It is the same `dataset_stats.pkl` produced during dataset preparation in the trainer half. See [trainer/docs/08_checkpoints_and_resume.md](../trainer/docs/08_checkpoints_and_resume.md) for where the trainer writes one.
2. Edit the JSON to point at it.

Without this, `RuntimeParams.read_stats_file()` (see [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py)) prints `File not found at: <path>` and returns `None`, which then propagates into the `DataNormalizationInterface` and crashes the first inference call.

The JSON's other keys (`HZ=20`, `proprio_state_dim=24`, `action_dim=24`, `action_chunk_size=50`, etc.) are correct for igris_b and need no edits unless you change hardware. See [04_configuration.md](04_configuration.md) for the full key list.

## 8. Start the Ray cluster

On each machine, in any order, run:

```bash
bash start_ray.sh
```

The script reads `$HOSTNAME` and chooses one branch:

| Hostname | What runs | Custom resource |
|---|---|---|
| `robros-MS-7E59` | `ray start --head` | `labeling_pc: 4` |
| `robros-ai1` | `ray start --address=$HEAD_IP:6379` | `training_pc: 3` |
| `robros-5090` | `ray start --address=$HEAD_IP:6379` | `inference_pc: 1` |

After all three are up, run `ray status` from any machine — you should see 3 nodes with the resource counts above. If you do not, jump to [09_troubleshooting.md](09_troubleshooting.md).

The `ulimit -n 65535` line in the `robros-ai1` branch raises the open-file-descriptor limit; the trainer holds many DataLoader workers and memmap files open at once, and the default Linux limit of 1024 is too low.

## 9. Launch a run

From any node that can reach the head (typically the head itself), with the venv activated:

```bash
python run_online_rl.py
```

That's it. The defaults in the argparse block of [run_online_rl.py](../run_online_rl.py) are wired for `igris_b` + `rtc` + the DSRL+OpenPI policy:

| Flag | Default |
|---|---|
| `--robot` | `igris_b` |
| `--inference_algorithm` | `rtc` |
| `--train_config` | `/home/user/Projects/online_rl/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml` |
| `--policy_yaml` | `./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml` |
| `--inference_runtime_params_config` | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json` |
| `--inference_runtime_topics_config` | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json` |
| `--num_labeler_gpus` | `4` (spawns 4 auto-labeler actors, each requesting 1 GPU) |
| `--human_reward_labeler` | off |

The default `--train_config` points at `/home/user/Projects/online_rl/...` — this absolute path almost certainly does not exist on your box. Pass `--train_config` explicitly, or change directory into your repo root and pass a path relative to it.

See [04_configuration.md](04_configuration.md) for what each config file controls.

## 10. Expected console output

The driver process logs roughly this sequence:

```text
running env_actor...
running training...
running 4 auto labelers...                                 # or "running human labeler..."
[from inference_pc] Warming up CUDA kernels...
[from inference_pc] Signaling inference ready...
[from inference_pc] Starting state readers...
[from inference_pc] Initializing robot position...
[from inference_pc] Control loop started...
[from training_pc] replay buffer size: 0
[from training_pc] replay buffer size: 0
...
[from training_pc] Replay buffer has been filled!
[from training_pc] Train iter: 0
[from training_pc] Train iter: 1
...
[from training_pc] Iteration: 0 -- Model: ... pushed from trainer
[from inference_pc] received weight from plasma...
[from inference_pc] Updating policy weights...
```

The trainer blocks until the replay buffer holds at least `batch_size * 2 * world_size` timesteps. You will see `replay buffer size: N` lines while episodes accumulate. Once the trainer starts iterating, weight pushes alternate with policy weight loads on the inference machine.

## First-run gotchas

- **`tailscale: command not found`** — install Tailscale on every machine (`curl -fsSL https://tailscale.com/install.sh | sh`). The script aborts if `tailscale ip -4` returns nothing.
- **`File not found at: /home/robros/.../dataset_stats.pkl`** — you skipped step 7.
- **`No module named trainer`** — submodules weren't initialised. Run `git submodule update --init --recursive`.
- **`Cannot find `transformers.models.gemma3...`** when loading the OpenPI policy — you skipped step 4. Rerun `bash openpi_transformer_lib_patch.sh`.
- **`Unknown robot: igris_c`** — the `igris_c` runtime files do not exist yet; only `--robot igris_b` works. See [09_troubleshooting.md](09_troubleshooting.md#--robot-igris_c-crashes-immediately).
- **`ManualRewardLabelerActor` never pulls from the queue** — known issue with `--human_reward_labeler`. The actor is created but its `start.remote()` is not called in that branch of [run_online_rl.py](../run_online_rl.py). See [09_troubleshooting.md](09_troubleshooting.md#manual-reward-labeler-never-pumps-the-queue).
- **`OSError: [Errno 24] Too many open files`** on the training machine — the `ulimit -n 65535` line in `start_ray.sh` only runs on the `robros-ai1` branch. If your training machine has a different hostname, add the `ulimit` line to your branch too.

Once you can complete one episode end-to-end and see at least one `Train iter:` line and one `weight pushed to Plasma...` line, the system is working. Move on to [02_architecture.md](02_architecture.md) to understand what just happened.
