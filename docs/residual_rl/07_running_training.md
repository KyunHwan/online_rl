← Back to [docs/residual_rl/README.md](./README.md)

# 07 — Running training (first-day quickstart)

This is the smallest set of steps to get residual RL running on your desk. It assumes you have access to the three-machine cluster described in [docs/03_distributed_setup.md](../03_distributed_setup.md). If you only have a single machine, see [Single-machine smoke test](#single-machine-smoke-test) at the bottom.

## Table of contents

- [Prereqs](#prereqs)
- [Clone, init, install](#clone-init-install)
- [Configure paths](#configure-paths)
- [Start Ray](#start-ray)
- [Smoke test — auto labeler, no teleop](#smoke-test--auto-labeler-no-teleop)
- [Full residual-RL run](#full-residual-rl-run)
- [Where outputs land](#where-outputs-land)
- [What "healthy" looks like](#what-healthy-looks-like)
- [Single-machine smoke test](#single-machine-smoke-test)

---

## Prereqs

| Requirement | Why |
|---|---|
| Linux (Ubuntu 22.04 tested) | The setup script `env_setup.sh` and `start_ray.sh` assume Debian/Ubuntu apt + bash. |
| Python 3.12.3 (managed by `uv`) | Pinned by `uv_setup.sh`. |
| Tailscale running on all three machines | `start_ray.sh` reads the Tailscale IP for cluster join. |
| ROS 2 Humble or later | The control loop uses `rclpy`. |
| At least one NVIDIA GPU per machine | bf16 autocast paths assume CUDA. |
| `git` with SSH access to `git@github.com:KyunHwan/online_rl` and `git@github.com:KyunHwan/trainer` | The submodule init pulls these. |

If you do not have the Tailscale-named hosts that `start_ray.sh` expects (`robros-MS-7E59`, `robros-ai1`, `robros-5090`), you will hit the `Unknown host` exit at the end of the case statement. See the bottom of the file for the commented-out single-machine fallback.

## Clone, init, install

```bash
git clone git@github.com:KyunHwan/online_rl.git
cd online_rl
git checkout features/residual_rl
git submodule update --init --recursive

# uv venv + uv pip
./uv_setup.sh
source .venv/bin/activate
./env_setup.sh
```

`env_setup.sh` installs PyTorch 2.9.0 + cu130, `torchrl`, `lerobot`, `wandb`, the OpenPI dependencies, `PySide6` for the manual labeler, plus the Robometer and Depth-Anything 3 submodules. It takes ~15 min on a fast connection.

## Configure paths

Two things need to be edited locally:

1. **`--train_config`** must point at an absolute path on the *training PC*. The default in [`run_online_rl.py:198-200`](../../run_online_rl.py#L198-L200) is `/home/user/Projects/online_rl/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml`. Change to your residual-RL config:

   ```
   <repo_root>/trainer/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml
   ```

2. **`train.save_dir`** inside the YAML you just chose ([resfit.yaml:65](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml#L65)) must be writable on the training PC. The default is `/home/user/Projects/online_rl/trainer/experiment_training/reinforcement_learning/resfit/online_rl`.

Optionally, set the wandb project name. `train_func` writes to a project named after `data.datamodule.params.task_name` (= `resfit_online_rl`).

## Start Ray

On each of the three machines, after activating the venv:

```bash
./start_ray.sh
```

`start_ray.sh` reads `hostname` and starts the right `ray start` invocation (head node on `robros-MS-7E59`, training-PC worker on `robros-ai1`, inference-PC worker on `robros-5090`). The resource quotas are now `{labeling_pc: 100, training_pc: 100, inference_pc: 100}`, so multi-actor placement does not block on quotas.

Confirm the cluster is up:

```bash
ray status
```

You should see three nodes with the right resource keys.

## Smoke test — auto labeler, no teleop

This is the fastest end-to-end loop to confirm the wiring without involving the master arm or pedal.

```bash
python run_online_rl.py \
  --robot igris_b \
  --train_config <abs path>/trainer/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml \
  --policy_yaml  ./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml \
  --residual_policy_yaml ./env_actor/policy/policies/resfit_policy/resfit_policy.yaml \
  --use_residual_rl \
  --inference_runtime_params_config ./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json \
  --inference_runtime_topics_config ./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json \
  --inference_algorithm rtc \
  --num_labeler_gpus 1
```

Expected stdout, in order:

```
running env_actor...
running 1 auto labelers...
Warming up CUDA kernels...
Starting state readers...
Starting control loop...
Waiting for inference actor to be ready...
running training...
Going into getting buffer...
replay buffer size: 0
replay buffer size: 0
...
Episode 0 finished!
Submitting episode 0 data...
replay buffer size: 256
Replay buffer has been filled!
Passed distributed barrier...
Entering training loop...
Got offline data batch
Got online data from replay buffer
Passed training loop barrier
Train iter: 0
```

The fill loop takes 1–2 episodes (about 25–50 s of robot time). Once you see `Train iter: 0`, the system is end-to-end alive.

You can `Ctrl-C` the driver to bring the run down. Ray cleans up the actors; the control-loop child processes shut down on the stop event.

## Full residual-RL run

The full run is identical to the smoke test but with `--num_labeler_gpus 4` and, optionally, teleop:

```bash
python run_online_rl.py \
  --robot igris_b \
  --train_config <abs path>/resfit.yaml \
  --policy_yaml  ./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml \
  --residual_policy_yaml ./env_actor/policy/policies/resfit_policy/resfit_policy.yaml \
  --use_residual_rl \
  --use_human_intervention \
  --inference_algorithm rtc \
  --num_labeler_gpus 4
```

With `--use_human_intervention`, you also need:

- The pedal driver running ([env_actor/human_in_the_loop/intervention_methods/pedal/publisher/pedal_publisher.py](../../env_actor/human_in_the_loop/intervention_methods/pedal/publisher/pedal_publisher.py)), publishing to `/igris_b/<robot_id>/io_event`.
- The Manus UDP receiver bound to the gloves.
- The Dynamixel U2D2 cable plugged in; the master arm reads 12 servos at 25 Hz.

Master-arm sync-read is now tolerant of 10 consecutive failures ([`arms_dynamixel.py:170-189`](../../env_actor/human_in_the_loop/teleoperation/robots/igris_b/arms_dynamixel.py#L170-L189)). If you see "Sync read RX failure" warnings but the run keeps going, you can leave it alone. If you see them stop and `Motor id=N did not respond to ping at baudrate` raise, check the cable.

## Where outputs land

- **Checkpoints**: under `train.save_dir/epoch_<N>/` as `resfit_residual_actor.pt`, `resfit_q_function.pt`, and the `*_opt.pt` companions ([online_trainer.py:299-338](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L299-L338)).
- **Wandb run**: project `resfit_online_rl`, run name `<task>_<train.project_name>` (`'resfit_online_rl'`). Metrics: `Residual Q Loss`, `Residual Q Value`, `resfit_q_function grad_norm`, `resfit_residual_actor grad_norm`, `epoch`.
- **Replay-buffer memmap**: `/tmp/online_rl_auto_data/` and (if `use_hil_buffer=True`) `/tmp/online_rl_hil_data/`. Wiped at startup.
- **Dataset stats**: `train.save_dir/dataset_stats.pkl` (written by the dataloader, rank 0 only).
- **Driver process stdout**: control-loop, inference-loop, and trainer prints all interleave here because `log_to_driver=True` in `ray.init(...)`. Use `ray logs` for per-worker access.

## What "healthy" looks like

The metric that matters most early on is the buffer fill rate and the gradient norms.

- `replay buffer size: …` should grow monotonically, reaching `2 × batch_size × world_size = 256` within the first few episodes.
- `resfit_q_function grad_norm` should be O(1–10) and decrease over the first few thousand iterations.
- `Residual Q Loss` should decrease and then level off; the absolute value depends on the reward labeler's range. There is no canonical target.
- `Residual Q Value` is the actor-step's negated loss; it should *rise* over training (the actor is maximizing Q).
- The control-loop log should not print "Episode queue full, dropping sub-episode" — if it does, the labelers cannot keep up.

The single failure pattern that will not show up in metrics: if all gradient norms are exactly 0 on the actor side, double-check `online_update: true` is set for `resfit_residual_actor` in your trainer YAML. Without that flag the trainer never pushes weights and the inference-side residual stays at the initial uniform noise.

## Single-machine smoke test

`start_ray.sh` does not have a single-machine mode out of the box, but you can do this manually:

```bash
ray start --head --port=6379 \
  --resources='{"labeling_pc": 100, "training_pc": 100, "inference_pc": 100}'
```

Then run the same driver command. Caveats:

- The DSRL-OpenPI checkpoint and the residual MLP will both compete for the same GPU. Reduce `data.batch_size` in the trainer YAML and `num_labeler_gpus 1` on the CLI.
- The HIL path needs real hardware (master arm + glove + pedal). For a single-machine smoke test, skip `--use_human_intervention`.

This is intended for "does the loop tick at all" verification, not for any quality metric.
