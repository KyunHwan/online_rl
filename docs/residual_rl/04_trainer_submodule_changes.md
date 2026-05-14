← Back to [docs/residual_rl/README.md](./README.md)

# 04 — Trainer submodule changes

## Table of contents

- [Submodule pointer status](#submodule-pointer-status)
- [Files that constitute residual RL inside the submodule](#files-that-constitute-residual-rl-inside-the-submodule)
- [`online_trainer.train_func` — what runs on each Ray worker](#online_trainertrain_func--what-runs-on-each-ray-worker)
- [`resfit_trainer` — the actor-critic loop](#resfit_trainer--the-actor-critic-loop)
- [`Critic_Trainer` — the Q-function update](#critic_trainer--the-q-function-update)
- [`Actor_Trainer` — the residual-actor update](#actor_trainer--the-residual-actor-update)
- [Residual actor / Q-function architectures](#residual-actor--q-function-architectures)
- [Dataset factory — `resfit_lerobot_dataset_factory`](#dataset-factory--resfit_lerobot_dataset_factory)
- [Integration contract with the parent repo](#integration-contract-with-the-parent-repo)

---

## Submodule pointer status

| | Submodule URL | SHA on `main` | SHA on `features/residual_rl` |
|---|---|---|---|
| `trainer/` | https://github.com/KyunHwan/trainer | `3ca051a256c9068f77b556df98f538d9a6185ccf` | `3ca051a256c9068f77b556df98f538d9a6185ccf` |
| `data_labeler/auto/models/robometer/` | https://github.com/KyunHwan/robometer | `a3d08d1f9821eb57154b3146477f2bd405cea283` | `a3d08d1f9821eb57154b3146477f2bd405cea283` |

**Both submodule pointers are identical between branches.** See [README — Caveats](./README.md#caveats). The trainer-side residual-RL code was merged into the submodule and then pulled forward onto `main`; this branch consumes it but does not bump the pointer.

The trainer-internal compare URL between the merge-base pointer and the current pointer (in case you want to inspect the trainer-only diff): `https://github.com/KyunHwan/trainer/compare/b162f07...3ca051a`.

Every file path below is given as a pinned link to the trainer at `3ca051a`. Never link to `main` or `HEAD` of the trainer — those rot.

## Files that constitute residual RL inside the submodule

Inside `trainer/` at SHA `3ca051a`:

| File (pinned URL) | Role |
|---|---|
| [`trainer/trainer/online_trainer.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py) | The `train_func` that Ray executes on each `TorchTrainer` worker. Same function the imitation-learning path uses; the loss specifics come from the trainer recipe. |
| [`experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml) | The experiment YAML that wires everything together: models, optimizers, dataloader, trainer recipe. |
| [`experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py) | `ResfitTrainer` — orchestrates one Q-step + (one-in-10) actor-step + Polyak target update. |
| [`experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py) | `Critic_Trainer` — n-step TD critic update. |
| [`experiment_training/components/trainer/reinforcement_learning/resfit/utils/actor_trainer.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/actor_trainer.py) | `Actor_Trainer` — residual-actor loss = `-Q(s, a_base + delta).mean()`. |
| [`experiment_training/components/dataloader/resfit_lerobot_data.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/dataloader/resfit_lerobot_data.py) | The offline dataset factory; produces LeRobot samples with the same `delta_timestamps` layout the buffer uses. |
| [`policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py) | `Residual_Actor` — Resnet34Group + preprocessor + 5-layer MLP + Tanh × 0.25. |
| [`policy_constructor/model_constructor/blocks/experiments/resfit/residual_q_function.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_q_function.py) | `Q_Function` — ensemble of 10 MLP critics with shared ResNet34 image encoders. |
| [`experiment_models/resfit/exp1/resfit_residual_actor.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_models/resfit/exp1/resfit_residual_actor.yaml) | Training-side YAML for the residual actor (same shape as the inference one). |
| [`experiment_models/resfit/exp1/resfit_q_function.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_models/resfit/exp1/resfit_q_function.yaml) | YAML for the Q-function. |

## `online_trainer.train_func` — what runs on each Ray worker

The function is `train_func(config_path: str)` ([`online_trainer.py:364`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L364)). Per-worker behavior, in order:

1. **Load config** from YAML and validate.
2. **Init wandb** on rank 0; project name comes from `data.datamodule.params.task_name` (= `'resfit_online_rl'`).
3. **Build trainer**: `_build_trainer` constructs models from `experiment_models/resfit/exp1/*.yaml`, optimizers from `component_optims`, the loss (currently `l2_loss`, although `resfit_trainer` ignores it), and instantiates the `resfit_trainer` recipe.
4. **Build dataloader** from `resfit_lerobot_dataset_factory` ([`online_trainer.py:210`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L210)).
5. **Wait for the replay buffer to fill**: `while size < 2 * batch_size * world_size: sleep(0.5)` ([`online_trainer.py:428-432`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L428-L432)).
6. **Training loop** ([`online_trainer.py:442`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L442)):
   - Pull one batch from the offline LeRobot dataloader.
   - Pull one batch from the online replay buffer.
   - **Mirror `base_policy_action`**: if the online batch has `base_policy_action`, copy the offline batch's `action` into a new `base_policy_action` key so the trainer can treat offline samples as "no residual needed" baselines.
   - Concatenate on `dim=0` after resizing online images to match offline image size.
   - Call `trainer.train_step(data, stats)`.
   - On rank 0, log to wandb; every `config.train.save_every * 25` iterations:
     - save checkpoints of all models and optimizers,
     - **for every model with `online_update=True`, push its CPU state dict to `policy_state_manager`**.

The last bullet is the critical contract: only `resfit_residual_actor` is configured with `online_update=True` ([resfit.yaml:13-15](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml#L13-L15)), so the Q-function never leaves the training PC.

## `resfit_trainer` — the actor-critic loop

`ResfitTrainer.train_step` ([`resfit_trainer.py:69`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py#L69)):

```
critic_loss   = Critic_Trainer(data, stats)           # every iter
critic_loss.backward(); clip; q_opt.step(); polyak_update(q_target, q, τ)

if iter > 0 and iter % 10 == 0:
    actor_loss = Actor_Trainer(data, stats)           # one in 10 iters
    actor_loss.backward(); clip; actor_opt.step()
```

Gradient clipping is `clip_grad_norm_(..., max_norm=1.0)` for every parameter that has an optimizer ([`resfit_trainer.py:134-146`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py#L134-L146)). Metric names written to wandb are `Residual Q Loss`, `resfit_q_function grad_norm`, `Residual Q Value`, `resfit_residual_actor grad_norm`.

## `Critic_Trainer` — the Q-function update

The critic is trained on **two timesteps from each chunked sample**: index `1` for the anchor state and index `3` for the future state ([`critic_trainer.py:32-46`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L32-L46)). The TD-target construction:

```
with torch.no_grad():
    a_base_t+3 = data["base_policy_action"][:, 3]
    s_t+3      = {head/left/right/proprio} at [:, 0]
    delta_t+3  = residual_actor(s_t+3, a_base_t+3)
    a_t+3      = delta_t+3 + a_base_t+3
    Q_target_t+3 = q_target(s_t+3, a_t+3, subsample_q=True)   # uses target Q (Polyak-averaged), random half of ensemble
    R_chunk    = Σ_{k=0..reward_horizon-1} γ^k · r[:, k]       # n-step discounted reward
    td_target  = R_chunk + γ^reward_horizon · Q_target_t+3

q_loss = MSE(q_function(s_t+1, a_t+1, critic=True), td_target)  # critic=True uses random half of ensemble
```

Two practical subtleties:

- The critic loss is computed against `data["action"]` (the executed action — base + residual), not against the base alone. This means the Q-function is learning the value of the *combined* policy.
- `subsample_q=True` on the target and `critic=True` on the online network both pick a *random* half of the 10-ensemble each step, regularizing against optimism bias ([`residual_q_function.py:236-264`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_q_function.py#L236-L264)). This pattern is reminiscent of TD3's twin critics, generalized to a larger ensemble.

`update_target` does Polyak averaging with `τ = 0.005` ([`critic_trainer.py:112-115`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L112-L115)).

## `Actor_Trainer` — the residual-actor update

The actor uses **only the anchor state** (index 1) and the anchor base action ([`actor_trainer.py:24-49`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/actor_trainer.py#L24-L49)):

```
delta      = residual_actor(s_t+1, a_base_t+1)        # gradients flow through residual
a_combined = delta + a_base_t+1
loss       = -1.0 * q_function(s_t+1, a_combined, subsample_q=False, critic=False).mean()
```

`q_function.eval()` is called first, so dropout/BN do not pollute the Q-value used for the actor target. Gradients only flow through the residual; the Q-function's parameters are *not* zeroed during the actor step because they have a separate optimizer (`resfit_q_function`'s optimizer is not stepped this iteration).

Hyper-parameters in [`resfit.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml):

- `resfit_residual_actor`: AdamW warm-cosine, `peak_lr=1e-4`, `total_steps=200000`, `warmup_steps=2000`.
- `resfit_q_function`: same schedule, `peak_lr=3e-4`.

## Residual actor / Q-function architectures

Both monolithic modules share a `Resnet34Group` of three ImageNet-pretrained ResNet-34 backbones (one per camera) followed by an `Eq.` 1×1 conv projection to 24 channels, flattened to 1 920 features per camera ([`residual_actor.py:12-55`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py#L12-L55), [`residual_q_function.py:14-57`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_q_function.py#L14-L57)). The preprocessor concatenates three flattened image features + the proprio history + the action chunk to form a 5 808-D vector.

| Hyper-parameter | Residual actor | Q-function |
|---|---|---|
| `input_dim` | 5 808 | 5 808 |
| Hidden width | 2 048 | 2 048 |
| Hidden depth | 5 layers | 4 layers |
| Activation | ELU | ELU |
| Norm | LayerNorm | LayerNorm |
| Ensemble size | 1 | 10 |
| Output | `Tanh(...)` × `0.25` (24-D residual) | mean of subsampled ensemble |
| Image preprocessing | resize to 240 × 320, ImageNet mean/std | same |

The Tanh × 0.25 ceiling is what makes the action bounded: the residual cannot exceed ±0.25 per dimension regardless of training dynamics. The bound is hard-coded in [`residual_actor.py:222`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py#L222) and is the easiest thing to change if you need a different correction magnitude — but be aware that the Q-function was trained against this scale, so changing it on the inference side without retraining will desynchronize the actor and the critic.

## Dataset factory — `resfit_lerobot_dataset_factory`

[`resfit_lerobot_data.py:11`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/dataloader/resfit_lerobot_data.py#L11). Loads a LeRobot dataset (`repo_id` defaults to `joon001001/igris-b-pnp-v4`) with `delta_timestamps` keyed by `dt = 1 / HZ`. The offset pattern for actions and rewards is `[0, dt, 2dt, …, (action_horizon-1)dt]`; for proprio and images it is the same pattern as the replay buffer (`action_horizon - 1, action_horizon - 4, …, 0` with a hard-coded step of `3`). Image augmentation is a `ColorJitter + GaussianBlur` Composite ([`resfit_lerobot_data.py:62-66`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/dataloader/resfit_lerobot_data.py#L62-L66)).

## Integration contract with the parent repo

The trainer expects these named Ray actors to exist at the time `train_func` starts:

| Ray actor name | Namespace | Producer | Consumer (trainer side) |
|---|---|---|---|
| `replay_buffer` | `online_rl` | [`run_online_rl.py:88-93`](../../run_online_rl.py#L88-L93) creates `ResfitReplayBufferActor` | [`online_trainer.py:421`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L421) calls `ray.get_actor("replay_buffer")` |
| `policy_state_manager` | `online_rl` | [`run_online_rl.py:84-85`](../../run_online_rl.py#L84-L85) creates `StateManagerActor` | [`online_trainer.py:424`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L424) pushes residual-actor weights here |

The replay buffer must speak two methods:

- `size() -> int` — used during the fill-wait warmup.
- `sample(batch_size: int) -> TensorDict` — returns batched chunked samples.

The keys the trainer reads from each sample are documented in `_pack_lerobot_like` ([`resfit_replay_buffer.py:136-180`](../../data_bridge/resfit_replay_buffer.py#L136-L180)) and consumed by the trainer in [`critic_trainer.py:24-71`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L24-L71):

| Key (replay buffer + offline) | Type | Shape |
|---|---|---|
| `action` | float32 | `(B, action_horizon, 24)` |
| `base_policy_action` | float32 | `(B, action_horizon, 24)` |
| `labels.reward` | float32 | `(B, reward_horizon)` |
| `observation.state` | float32 | `(B, |proprio_offsets|, 24)` |
| `observation.current` | float32 | same as `observation.state` |
| `observation.images.cam_{head,left,right}` | uint8 → float32 | `(B, |image_offsets|, 3, H, W)` |

If you change the replay buffer's offset construction, every one of those shapes must continue to match what the trainer reads — that is the silent contract between the two repos.
