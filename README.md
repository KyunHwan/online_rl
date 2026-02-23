# Online RL

A distributed online reinforcement learning framework for robotic manipulation. The system runs real-time policy inference on a robot, collects episodes, labels rewards (automatically via a VLM or manually via a GUI), stores experience in a replay buffer, and trains the policy online — all orchestrated as a [Ray](https://docs.ray.io/) cluster spanning multiple machines.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      Ray Cluster (3 machines)                     │
│                                                                   │
│  inference_pc             training_pc            labeling_pc      │
│                                                                   │
│  ┌────────────┐        ┌───────────────┐      ┌──────────────┐   │
│  │  EnvActor  │──ep───>│ RewardLabeler │──────>│ ReplayBuffer │   │
│  │ (RTCActor) │ queue  │  (VLM / GUI)  │      │  (memmap)    │   │
│  └────────────┘        └───────────────┘      └──────┬───────┘   │
│        ▲                                             │ sample()   │
│        │ weights                              ┌──────▼───────┐   │
│   ┌────┴──────┐                               │   Trainer    │   │
│   │   State   │<──────────────────────────────│  (Ray DDP)   │   │
│   │  Manager  │           weights             └──────────────┘   │
│   └───────────┘                                                   │
└───────────────────────────────────────────────────────────────────┘
```

The episode data flow is: **EnvActor** produces episodes and enqueues them via a Ray Queue. The **RewardLabeler** dequeues episodes, annotates rewards, and writes the labeled data to the **ReplayBuffer**. The **Trainer** samples batches from the replay buffer and pushes updated weights to the **StateManager**, which the EnvActor polls for new weights.

**Data-format boundary** — outside the policy everything is numpy; torch lives only inside the policy:

```
Robot ──numpy obs──▶ Policy [ normalize(numpy) → torch → inference → numpy ] ──numpy actions──▶ Robot
```

## Key Concepts

### Policy Protocol

Every policy implements the protocol defined in [`env_actor/policy/templates/policy.py`](env_actor/policy/templates/policy.py):

| Method | Purpose |
|--------|---------|
| `predict` | Single forward pass; returns `np.ndarray` `(action_horizon, action_dim)` |
| `guided_inference` | Action-inpainting inference; blends previous chunk with new prediction |
| `warmup` | Triggers CUDA kernel warm-up (`torch.backends.cudnn.benchmark`) |
| `freeze_all_model_params` | Freezes all parameters for inference-only mode |

A policy is composed of model **components** (each an `nn.Module`) and exposes them via `self.components`. The trainer uses `self.components` to push updated weights after each training round.

### Normalization Manager

The [`DataNormalizationInterface`](env_actor/nom_stats_manager/data_normalization_interface.py) performs input normalization **using only numpy** — no torch dependency. It is passed **into** the policy at inference time so that normalization happens inside the policy call. The normalizer uses dataset statistics (mean/std) loaded from a pickle file:

- Proprioceptive state: `(value - mean) / (std + eps)`
- Camera images: `value / 255.0`

### Autonomous Inference (`env_actor/auto`)

Fully autonomous policy execution without human intervention. Two inference algorithms are available:

- **RTC (Real-Time Action Chunking)** — the default. Spawns two OS-level processes that communicate via shared memory: one runs GPU inference, the other handles the robot control loop at a fixed frequency.
- **Sequential** — a single-threaded synchronous loop: read state → infer → publish action.

### Guided Inference / Action Inpainting

Based on [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339). When the policy predicts a new action chunk, it blends the previously un-executed actions with the new prediction using exponentially decaying weights. This compensates for GPU inference latency in real-time control. The blending logic lives inside the policy's `guided_inference()` method and uses [`compute_guided_prefix_weights()`](env_actor/inference_engine_utils/action_inpainting.py).

### Data Format Boundary

| Location | Format |
|----------|--------|
| Robot I/O, data managers, normalization manager | **numpy** (`float32` proprio, `uint8` images) |
| Inside the policy (forward pass) | **torch** tensors (converted internally) |
| Policy output (actions) | **numpy** `float32` |

Torch tensors never leak outside the policy boundary.

## Repository Layout

| Directory | Description |
|-----------|-------------|
| [`env_actor/`](env_actor/) | Environment-side logic: policy inference, robot I/O, data recording, normalization. [README](env_actor/README.md) |
| [`data_bridge/`](data_bridge/) | Replay buffer (disk-backed via memmap) and state manager for weight distribution. [README](data_bridge/README.md) |
| [`data_labeler/`](data_labeler/) | Reward labeling — automatic (VLM) or manual (PySide6 GUI). [README](data_labeler/README.md) |
| `trainer/` | Training library (git submodule). Online training loop, model construction, registry system. See [trainer/README.md](trainer/README.md) |

**Top-level files:**

| File | Purpose |
|------|---------|
| [`run_online_rl.py`](run_online_rl.py) | Main entrypoint — orchestrates all Ray actors |
| [`run_online_rl_openpi.py`](run_online_rl_openpi.py) | OpenPI-specific variant of the entrypoint |
| [`start_ray.sh`](start_ray.sh) | Starts the Ray cluster on each machine |
| [`env_setup.sh`](env_setup.sh) | Installs all Python dependencies via `uv pip` |
| [`uv_setup.sh`](uv_setup.sh) | Initializes the `uv` package manager |

## Quickstart

### 1. Install Dependencies

```bash
bash uv_setup.sh      # set up uv package manager
bash env_setup.sh     # install PyTorch, Ray, and all other dependencies
```

### 2. Start the Ray Cluster

Run on **each machine** in the cluster. The script uses hostname-based routing to assign Ray resources:

```bash
bash start_ray.sh
```

The script assigns custom resources per machine:
- `training_pc` — head node (runs trainer + replay buffer + state manager)
- `inference_pc` — inference node (runs the env actor with GPU)
- `labeling_pc` — labeling node (runs the reward labeler)

### 3. Run the Online RL Loop

```bash
python run_online_rl.py \
  --robot igris_b \
  --inference_algorithm rtc \
  --train_config /path/to/train_config.yaml \
  --policy_yaml ./env_actor/policy/policies/openpi_policy/openpi_policy.yaml \
  --inference_runtime_params_config /path/to/inference_runtime_params.json \
  --inference_runtime_topics_config /path/to/inference_runtime_topics.json
```

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--robot` | *(required)* | Robot identifier: `igris_b` or `igris_c` |
| `--inference_algorithm` | `rtc` | `rtc` or `sequential` |
| `--train_config` | see below | Absolute path to training config YAML |
| `--policy_yaml` | see below | Path to policy config YAML |
| `--inference_runtime_params_config` | see below | Absolute path to runtime params JSON |
| `--inference_runtime_topics_config` | see below | Absolute path to runtime topics JSON |
| `--human_reward_labeler` | `False` | Use manual GUI labeler instead of VLM |

**Defaults for config paths:**
- `--train_config`: `trainer/experiment_training/imitation_learning/vfp_single_expert/exp2/vfp_single_expert_depth.yaml`
- `--policy_yaml`: `./env_actor/policy/policies/openpi_policy/openpi_policy.yaml`
- `--inference_runtime_params_config`: `env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json`
- `--inference_runtime_topics_config`: `env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json`

The alternate entrypoint [`run_online_rl_openpi.py`](run_online_rl_openpi.py) accepts the same flags and is tailored for OpenPI policy configurations.

## Configuration

The system uses three layers of configuration:

### Training Config (YAML)

Passed via `--train_config`. Defines the training loop, model architecture, data loading, loss function, optimizer, and checkpointing. Parsed by the trainer's config system. Example configs live under `trainer/experiment_training/`.

### Policy Config (YAML)

Passed via `--policy_yaml`. Specifies which model components to build and which policy class to instantiate. Example: [`env_actor/policy/policies/openpi_policy/openpi_policy.yaml`](env_actor/policy/policies/openpi_policy/openpi_policy.yaml):

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: openpi_policy
```

The `component_config_paths` are resolved relative to the YAML file's directory. The `policy.type` must match a key registered in `POLICY_REGISTRY`.

### Runtime Params / Topics (JSON)

Passed via `--inference_runtime_params_config` and `--inference_runtime_topics_config`. Robot-specific parameters controlling:

- Control frequency (`HZ`), policy update period
- Proprioceptive state dimension and history size
- Camera names, image dimensions, observation count
- Action dimension and chunk size
- Path to normalization statistics file
- ROS topic names for robot communication

Per-robot JSON configs live under [`env_actor/runtime_settings_configs/robots/`](env_actor/runtime_settings_configs/).

## Distributed Setup

The [`start_ray.sh`](start_ray.sh) script uses [Tailscale](https://tailscale.com/) for cross-machine networking. Each machine is identified by hostname and assigned a custom Ray resource label:

| Hostname | Resource | Role |
|----------|----------|------|
| `robros-ai1` | `training_pc: 3` | Head node — trainer, replay buffer, state manager |
| `robros-5090` | `inference_pc: 1` | Inference node — RTCActor (GPU) |
| `robros-MS-7E59` | `labeling_pc: 1` | Labeling node — reward labeler |

Ray actors are pinned to machines via `resources={"<resource>": 1}` in their `.options()` calls.

## Extending the System

### Adding a New Policy

1. Create a directory under `env_actor/policy/policies/<your_policy>/`.
2. Implement the [`Policy`](env_actor/policy/templates/policy.py) protocol.
3. Register your class with `@POLICY_REGISTRY.register("your_policy")`.
4. Create a policy YAML that sets `policy.type: your_policy` and lists your model component configs under `model.component_config_paths`.

The loader ([`env_actor/policy/utils/loader.py`](env_actor/policy/utils/loader.py)) will auto-import your module and build the policy.

### Adding a New Robot

Each robot requires a set of bridge implementations under per-directory `robots/<robot_name>/` folders:

| Directory | What to add |
|-----------|-------------|
| `env_actor/robot_io_interface/robots/` | `controller_bridge.py` — read state, publish actions |
| `env_actor/nom_stats_manager/robots/` | `data_normalization_manager.py` |
| `env_actor/episode_recorder/robots/` | `episode_recorder_bridge.py` |
| `env_actor/auto/.../rtc/data_manager/robots/` | `shm_manager_bridge.py` |
| `env_actor/auto/.../sequential/data_manager/robots/` | `data_manager_bridge.py` |
| `env_actor/runtime_settings_configs/robots/` | `inference_runtime_params.py` + JSON |

Then add your robot's import branches to the existing interface classes (e.g., `ControllerInterface`, `DataNormalizationInterface`).

### Adding a New Inference Algorithm

1. Create a directory under `env_actor/auto/inference_algorithms/<your_algo>/`.
2. Implement a Ray actor class with a `start()` method.
3. Add the algorithm choice to the `--inference_algorithm` argparse in [`run_online_rl.py`](run_online_rl.py).

### Invariants to Respect

When extending any part of the system, these invariants must hold:

1. Normalization is done **inside** the policy. The policy receives a `DataNormalizationInterface` as input.
2. Policy exposes model components via `self.components` (used for weight updates).
3. Guided inference (action inpainting) happens **inside** the policy.
4. All data entering the policy is **numpy arrays**.
5. The normalization manager uses **only numpy** (no torch).
6. Policy inference output is a **numpy array**.
7. Torch tensor computation happens **only inside** the policy.
