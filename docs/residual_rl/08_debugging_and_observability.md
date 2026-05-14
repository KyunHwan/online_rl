← Back to [docs/residual_rl/README.md](./README.md)

# 08 — Debugging and observability

## Table of contents

- [Metrics](#metrics)
- [Log lines: signal vs noise](#log-lines-signal-vs-noise)
- [Common failure modes](#common-failure-modes)
- [Running components in isolation](#running-components-in-isolation)
- [Tracing tensors safely under Ray + multiprocessing](#tracing-tensors-safely-under-ray--multiprocessing)

---

## Metrics

Wandb metrics emitted by `resfit_trainer` (one per `train_step`, rank 0 only):

| Metric | Source | Meaning | Healthy direction |
|---|---|---|---|
| `Residual Q Loss` | [`Critic_Trainer.forward`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L75-L86) | MSE between Q(s, a) and the n-step TD target. | ↓ then plateau |
| `Residual Q Value` | [`Actor_Trainer.forward`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/actor_trainer.py#L43-L49) | Negated actor loss = `Q(s, a_base + delta).mean()`. | ↑ over training |
| `resfit_q_function grad_norm` | [`_clip_get_grad_norm`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py#L134-L146) | L2 of Q's gradient before clip. | O(1–10), declining |
| `resfit_residual_actor grad_norm` | same | L2 of actor's gradient before clip. | O(0.1–10), declining |
| `epoch` | [`_record`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L343-L355) | `iterations / num_iter_per_epoch`. Used for plotting. | ↑ monotonically |

There are **no per-step environment-reward metrics**. If you want one, add a wandb log inside the control loop or inside the labeler — neither currently emits one.

## Log lines: signal vs noise

The driver process collects logs from every Ray worker because `log_to_driver=True` is set in `ray.init` ([`run_online_rl.py:71-77`](../../run_online_rl.py#L71-L77)). The volume is high — focus on:

### Signal (watch these)

| Substring | From | Means |
|---|---|---|
| `replay buffer size: <N>` | trainer worker | Fill progress before training begins. |
| `Train iter: <N>` | trainer worker | One critic step landed. |
| `Iteration: <N> -- Model: resfit_residual_actor pushed from trainer` | trainer worker | New residual weights are on the wire to inference. |
| `Updating policy weights...` then `resfit_residual_actor weights updated` | control loop | Inference side picked up the new weights at the next episode boundary. |
| `Episode <N> finished!` | control loop | An episode actually completed (1000 steps or stop event). |
| `Submitting episode <N> data...` | control loop | Sub-episodes are entering the labeler queue. |
| `Episode queue full, dropping sub-episode from episode <N>` | control loop | **Bad.** Labeler cannot keep up. Look at labeler logs next. |
| `Sync read RX failure (code <N>), consecutive=<K>` | teleop master arm | <K=10 is recoverable; ≥10 raises. |
| `Motor id=N did not respond to ping ...` | teleop master arm | Cable / power issue. Won't auto-recover. |
| `TRAINING ERROR at iteration <N>: ...` | trainer worker | Training raised. Stack trace follows. |
| `[PedalSwitch] -> TELEOP` / `-> POLICY` | pedal subscriber | Pedal mode changed. |
| `Control Mode: TELEOP !!!!!!!!!!` / `Control Mode: POLICY` | HIL control loop | Per-step indication of which source produced the action. |

### Noise (do not chase these)

| Substring | Why ignore |
|---|---|
| `Warmup encountered error (may be expected for minimal inputs)` | The policy's `warmup()` swallows expected exceptions for tiny dummy inputs ([`inference_loop.py:77-80`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L77-L80)). |
| `WARNING: Could not set multiprocessing start method` | Only happens on re-runs in the same Python session ([`run_online_rl.py:12-15`](../../run_online_rl.py#L12-L15)). |
| `Teleop Action is None !!!` | Master arm has not yet emitted a sample; happens once per episode if you start in TELEOP mode. |
| `Going into getting buffer...` looping | Initial fill is naturally slow on first run. |

## Common failure modes

### NaN loss / NaN grad_norm

**Cause**: most commonly the reward labeler is emitting `nan` for some episodes, which propagates into `data["labels.reward"]` and then into the TD target.

**Diagnose**: in the trainer, after `online_data = ray.get(future)`, log `online_data['labels.reward'].isnan().any().item()`. Alternatively, gate the trainer's logging on `torch.isnan(loss)`.

**Fix**: filter at the labeler. The labeler's `add()` call into the buffer is the right cut point — if the reward is `nan`, drop the sub-episode.

### `replay buffer size:` stuck at 0

**Cause(s)**:
1. The labeler is not connected to the queue. Check `running 4 auto labelers...` or `running human labeler...` printed at startup.
2. The control loop is producing sub-episodes but the labeler crashed silently — look in `ray logs` for the labeler actor.
3. The control loop is not finishing episodes — check whether `Episode 0 finished!` ever prints.

**Fix path**: from the leaf upward. Make sure the control loop hits episode end (1000 steps at 40 Hz is ~25 s).

### `Episode queue full, dropping sub-episode`

**Cause**: labelers cannot label as fast as the control loop produces. Each auto-labeler GPU does ~real-time labeling; if you set `--num_labeler_gpus 1` but the control loop is running at full speed, you will drop episodes.

**Fix**: increase `--num_labeler_gpus`, or accept the drops if you are throughput-limited on the labeler GPU.

### Trainer stuck before `Train iter: 0`

**Cause**: replay buffer reached the threshold but the offline LeRobot dataset is hanging. Most often this is a Hugging Face download in the first run — the dataloader is `not local_files_only`, so it goes online to fetch `joon001001/igris-b-pnp-v4`.

**Fix**: pre-download. Set `data.datamodule.params.local_files_only: true` and `data.datamodule.params.root: <path>` in the trainer YAML once the dataset is on disk.

### Inference loop never publishes a chunk

**Cause**: the inference loop is blocked in `wait_for_min_actions(35)`. That means `num_control_iters` is not advancing — the control loop is stuck before its first action.

**Diagnose**: look for `Starting control loop...` in the driver output. If you see `Waiting for inference actor to be ready...` but then nothing, the handshake failed — check that `policy.warmup()` did not raise an unrecoverable error.

### Residual weights never refresh

**Symptoms**: control loop logs `Updating policy weights...` once at startup, then nothing for hours, and the trainer logs `Iteration: N -- Model: ... pushed from trainer` is missing.

**Cause**: `online_update: true` is missing on `resfit_residual_actor` in your trainer YAML. The push step is gated on that flag in [`online_trainer.py:537`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L537).

### Shared-memory cleanup warning at shutdown

Lines like `FileNotFoundError` during shm unlinking are expected and swallowed by [`rtc_actor.py:172-187`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L172-L187). The unlink can race with the OS resource_tracker.

### Replay buffer reads `RuntimeError: requested shape ... doesn't match storage`

**Cause**: the buffer's offsets and the dataloader's `delta_timestamps` disagree. Check that `action_horizon`, `reward_horizon`, and the proprio/image subsample pattern match between [`run_online_rl.py:88-93`](../../run_online_rl.py#L88-L93) and the trainer YAML.

### Dynamixel sync-read keeps failing

**Mitigated automatically** up to 10 consecutive failures. If it exceeds that, look at:
- USB cable seating.
- Power supply voltage (these motors stall under brown-out).
- The motor IDs listed in the raised error message (`unresponsive ids=[...]`).

## Running components in isolation

### Run only the labeler

Spawn a `RayQueue` and a `ResfitReplayBufferActor` named `replay_buffer`, then run:

```python
import ray
ray.init(address="auto", namespace="online_rl")
from data_labeler.auto.auto_reward_labeler import AutoRewardLabelerActor
queue = ...  # named queue
buffer = ...  # ResfitReplayBufferActor handle
lab = AutoRewardLabelerActor.options(resources={"labeling_pc": 1}, num_gpus=1).remote(
    episode_queue_handle=queue,
    replay_buffer_actor=buffer,
    img_frame_key="head",
    reward_key="reward",
)
ray.get(lab.start.remote())
```

You can then feed synthetic episodes into the queue and check that `buffer.size.remote()` advances.

### Run only the RTC actor

Skip `run_online_rl.py` and call `RTCActor` directly. It needs:

- a `policy_state_manager` Ray actor (the `data_bridge.state_manager.StateManagerActor`),
- a `RayQueue` for sub-episodes (can be a no-op sink),
- both the base and residual policy YAMLs.

The actor will run the rollout loop and submit sub-episodes to whatever queue you give it. Useful for isolating control/inference timing.

### Run only the trainer

Manually create the `replay_buffer` and `policy_state_manager` Ray actors first, then call `train_func(config_path)` directly:

```python
from trainer.trainer.online_trainer import train_func
train_func("<abs path>/resfit.yaml")
```

`train_func` is normally invoked by `TorchTrainer.fit()` so a Ray context is expected. To run it locally for unit-debugging, wrap it in a `TorchTrainer` with `num_workers=1, use_gpu=True`.

## Tracing tensors safely under Ray + multiprocessing

Two pitfalls when adding `print(tensor)` for debugging:

1. **Do not print CUDA tensors inside Ray remotes** without first calling `.detach().cpu()` — under certain Ray serialization paths, the print can block on the CUDA stream and you will get confusing-looking deadlocks.
2. **Do not log inside the shared-memory lock** ([`shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)). If the labeler is slow and you have a logger that goes through a `multiprocessing.Queue`, you can wedge the whole control loop.

A pragmatic pattern that works inside the control loop:

```python
if iter_count % 100 == 0:
    print(f"[ctrl] base={base_policy_action[0, :4]} residual={residual_action[0, :4]} t={time.perf_counter():.3f}")
```

Both arrays are already numpy at that point (`base_policy_action.copy()` on [`control_loop.py:198`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L198) and the residual is already cast back to numpy in [`resfit_policy.py:114`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py#L114)).
