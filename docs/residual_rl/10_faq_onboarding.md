← Back to [docs/residual_rl/README.md](./README.md)

# 10 — FAQ for onboarding

The questions a junior is likely to ask on days 2–5, with citation-backed answers.

## Table of contents

- [Q1. Why are/were there two `run_online_*.py` scripts?](#q1-why-arewere-there-two-run_online_py-scripts)
- [Q2. Where is the base policy actually loaded?](#q2-where-is-the-base-policy-actually-loaded)
- [Q3. Where is the residual policy actually loaded?](#q3-where-is-the-residual-policy-actually-loaded)
- [Q4. Can I train on CPU?](#q4-can-i-train-on-cpu)
- [Q5. How do I add a new reward term?](#q5-how-do-i-add-a-new-reward-term)
- [Q6. How do I swap the residual network architecture?](#q6-how-do-i-swap-the-residual-network-architecture)
- [Q7. What's the relationship between `data_bridge` and `data_labeler`?](#q7-whats-the-relationship-between-data_bridge-and-data_labeler)
- [Q8. How do I resume from a checkpoint?](#q8-how-do-i-resume-from-a-checkpoint)
- [Q9. The trainer pushes weights every `save_every × 25` iterations. Why the `× 25`?](#q9-the-trainer-pushes-weights-every-save_every-25-iterations-why-the-25)
- [Q10. What does `online_update: true` actually do?](#q10-what-does-online_update-true-actually-do)
- [Q11. Why is `run_training` not a Ray task anymore?](#q11-why-is-run_training-not-a-ray-task-anymore)
- [Q12. Why does the buffer wipe `/tmp/online_rl_auto_data/` at startup?](#q12-why-does-the-buffer-wipe-tmponline_rl_auto_data-at-startup)
- [Q13. What is `obs_subsample_step` and why is it 3?](#q13-what-is-obs_subsample_step-and-why-is-it-3)
- [Q14. The buffer's `add()` has a `separate_key` argument — what's it for?](#q14-the-buffers-add-has-a-separate_key-argument--whats-it-for)
- [Q15. How do I get the residual to start from random instead of zeros?](#q15-how-do-i-get-the-residual-to-start-from-random-instead-of-zeros)
- [Q16. The action mux uses shortest-arc interpolation for joints. Why?](#q16-the-action-mux-uses-shortest-arc-interpolation-for-joints-why)

---

## Q1. Why are/were there two `run_online_*.py` scripts?

There used to be `run_online_rl.py` and `run_online_rl_openpi.py`. The second was deleted on both branches (commit `b9369a4` on `features/residual_rl`, commit `01e9219` on `main`) because the OpenPI-specific entry point became a strict subset of `run_online_rl.py` once the policy YAML mechanism stabilized. There is now exactly one driver script.

## Q2. Where is the base policy actually loaded?

Inside the **inference loop child process**, at [`inference_loop.py:67-72`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L67-L72):

```python
policy = build_policy(
    policy_yaml_path=policy_yaml_path,
    map_location="cpu",
).to(device)
policy.eval()
```

`policy_yaml_path` is the value of `--policy_yaml`. For DSRL-OpenPI the actual checkpoint file lives in `params.ckpt_dir` of [`openpi_model.yaml`](../../env_actor/policy/policies/dsrl_openpi_policy/components/openpi_model.yaml). The policy YAML is loaded once per process; if you change it at runtime nothing happens.

## Q3. Where is the residual policy actually loaded?

Inside the **control loop child process**, at [`control_loop.py:101-105`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L101-L105):

```python
residual_policy = build_policy(
    policy_yaml_path=residual_policy_yaml_path,
    map_location="cpu",
).to(device)
residual_policy.eval()
```

If `resfit_policy.yaml: policy.params.checkpoint_path` is set, that path's `resfit_residual_actor.pt` is loaded into the component at construction ([`resfit_policy.py:52-56`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py#L52-L56)). Otherwise the residual starts from Xavier-init random.

## Q4. Can I train on CPU?

In practice, no. The bf16 autocast paths in the inference loop and trainer assume CUDA. The ResNet-34 forward pass for three cameras + a 2048-wide 5-layer MLP × 10 Q-ensemble at `batch_size=128` requires real GPU memory.

For very small smoke tests you could pin the runtime to CPU by editing `device = torch.device("cpu")` in [`control_loop.py:97`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L97) and [`inference_loop.py:63`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L63), and removing the `torch.autocast` wrappers. That is not a path the repo supports — it is a "unstuck the import" workaround for development only.

## Q5. How do I add a new reward term?

The reward is computed by the **labeler**, not by anything on this branch. To add a term:

1. Open the labeler implementation in [`data_labeler/auto/auto_reward_labeler.py`](../../data_labeler/auto/auto_reward_labeler.py).
2. Modify the per-step reward emitted into the sub-episode's TensorDict before it is forwarded to `replay_buffer.add(...)`.
3. The trainer will pick it up automatically because `labels.reward` is read from the buffer sample (see [`critic_trainer.py:60-71`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L60-L71)).

No change to the residual-RL code is needed if you preserve the schema (`labels.reward` is a `(reward_horizon,)` float tensor per sample).

## Q6. How do I swap the residual network architecture?

Two places must change in lockstep:

1. The **inference-side** YAML at [`components/resfit_residual_actor.yaml`](../../env_actor/policy/policies/resfit_policy/components/resfit_residual_actor.yaml).
2. The **trainer-side** YAML at [`experiment_models/resfit/exp1/resfit_residual_actor.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_models/resfit/exp1/resfit_residual_actor.yaml).

Both load through the same factory and expect the same `_type_: resfit_residual_actor` block. If you change `num_layers` or `num_hidden_dim` in one but not the other, the runtime `load_state_dict` call ([`control_loop.py:122-125`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L122-L125)) will raise on the first weight push because the keys won't match.

For a more invasive change (e.g. swap the MLP for a transformer), edit [`policy_constructor/.../residual_actor.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py) and register the new class under a new name. Then update both YAMLs to reference the new key.

## Q7. What's the relationship between `data_bridge` and `data_labeler`?

- [`data_bridge/`](../../data_bridge/) — Ray actors that bridge **data through the cluster**: the replay buffer (`ResfitReplayBufferActor`) and the weight broker (`StateManagerActor`). These are passive plumbing.
- [`data_labeler/`](../../data_labeler/) — actors that **consume episodes from the queue and produce rewards**. Two implementations: an auto labeler that uses a model (Robometer submodule) and a manual PySide6 labeler. These produce the `reward` field that the trainer reads.

`data_bridge` is shared between offline and online; `data_labeler` is online-only.

## Q8. How do I resume from a checkpoint?

For the **trainer**: set `train.load_dir` in your trainer YAML to the absolute path of the `epoch_<N>` directory under your `save_dir`. `_build_models` loads `<load_dir>/<key>.pt` for each model in the config ([online_trainer.py:142-147](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L142-L147)).

For the **inference-side residual**: set `policy.params.checkpoint_path` in [`resfit_policy.yaml`](../../env_actor/policy/policies/resfit_policy/resfit_policy.yaml) to a directory containing `resfit_residual_actor.pt`. The `ResidualPolicy.__init__` does the load ([`resfit_policy.py:52-56`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py#L52-L56)).

There is no "resume from rollout" mechanism — the replay buffer is always rebuilt from scratch at process start (see [Q12](#q12-why-does-the-buffer-wipe-tmponline_rl_auto_data-at-startup)).

## Q9. The trainer pushes weights every `save_every × 25` iterations. Why the `× 25`?

Look at [`online_trainer.py:526`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L526):

```python
if (iterations + 1) % (config.train.save_every * 25) == 0:
    _save_checkpoints(...)
    weights_ref = ray.put(policy_components_weights)
    policy_state_manager.update_state.remote(weights_ref)
```

`save_every` was the imitation-learning epoch cadence. The `× 25` is an empirical scaling so that, with the residual-RL setup's `save_every: 20`, the actual push happens every 500 iterations. That works out to roughly one push per few hundred rollout steps. Treat the `25` as a magic number; if you want a faster push, lower `save_every` rather than touching the multiplier (the multiplier rotates with the imitation-learning code path that shares this function).

## Q10. What does `online_update: true` actually do?

It is a per-model boolean in `model.component_build_args` that gates two things at training time:

1. **Whether the trainer pushes the model's weights**: see [`online_trainer.py:537`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L537):
   ```python
   if not config.model.component_build_args[model_name]['freeze'] and config.model.component_build_args[model_name]['online_update']:
       ...
       policy_state_manager.update_state.remote(weights_ref)
   ```
2. *(That is the only effect.)*

On this branch, `resfit_residual_actor: online_update: true` and `resfit_q_function: online_update: false`. So the Q-function stays on the training PC and the residual actor's weights are shipped to the inference side.

## Q11. Why is `run_training` not a Ray task anymore?

In `main`, `run_training` was decorated `@ray.remote` and launched as `run_training.options(...).remote(...)` plus a `ray.get(train_ref)` later. On this branch the decorator is commented out and the function is called synchronously from the driver ([`run_online_rl.py:175-177`](../../run_online_rl.py#L175-L177)).

The likely reason (commits `4cc3b5d removed ray actor` and `60cb392 remvoed ray.remote for run_training`): the Ray-remote wrapping was double-distributing inside `TorchTrainer.fit()` (which itself spawns Ray workers via `ScalingConfig`), so there was no benefit, only the cost of an extra Ray worker that does little besides call `.fit()`.

This is one of the "verify with maintainer" items listed in [00_overview.md](./00_overview.md#verify-with-maintainer) — confirm the choice if you are planning to re-enable distributed scheduling on the training PC.

## Q12. Why does the buffer wipe `/tmp/online_rl_auto_data/` at startup?

`LazyMemmapStorage` requires `existsok=True` to reuse an existing directory, but past runs' memmap files can have stale shapes (e.g. you bumped `capacity` from 100k to 200k). The buffer's constructor explicitly `shutil.rmtree`s the scratch dir on every start ([`resfit_replay_buffer.py:53-55`](../../data_bridge/resfit_replay_buffer.py#L53-L55)) so this class of "shape mismatch with stale tmp files" failure cannot bite.

This means **the replay buffer is not persistent across runs**. Each start collects fresh rollouts.

## Q13. What is `obs_subsample_step` and why is it 3?

Default in `run_online_rl.py` is `obs_subsample_step=3`. It controls the stride of proprio/image offsets in `_build_offsets` ([`resfit_replay_buffer.py:91-95`](../../data_bridge/resfit_replay_buffer.py#L91-L95)):

```python
proprio_offsets = torch.arange(action_horizon - 1, -1, -obs_subsample_step, dtype=torch.long)
image_offsets   = torch.arange(action_horizon - 1, -1, -obs_subsample_step, dtype=torch.long)
```

With `action_horizon=4` this yields `[3, 0]`. The Critic_Trainer uses index 1 as the anchor state ("now") and index 0 as the future state. Setting `obs_subsample_step=1` would give `[3, 2, 1, 0]`, allowing the critic to see more history at the cost of more images per sample.

The `3` is also hard-coded in the offline dataloader's `obs_proprio_timestamps` ([resfit_lerobot_data.py:47-48](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/dataloader/resfit_lerobot_data.py#L47-L48)) — if you change the buffer, change the dataloader too.

## Q14. The buffer's `add()` has a `separate_key` argument — what's it for?

When `use_hil_buffer=True`, the buffer splits incoming episodes into two storages based on the value of `tensordict[separate_key][0]`. The default `separate_key="control_mode"` looks at `0 == POLICY` and `1 == TELEOP`. Auto rollouts (no teleop, `control_mode=0`) go to the main buffer; teleop-interrupted rollouts go to the HIL buffer ([`resfit_replay_buffer.py:122-130`](../../data_bridge/resfit_replay_buffer.py#L122-L130)).

`sample()` then pulls half from each. The point is to avoid drowning the off-policy critic in pure-teleop data when an operator is dragging the robot through demonstrations.

This path is currently unused in `run_online_rl.py` — the actor is instantiated with defaults, which means `use_hil_buffer=False`. To turn it on, change the `.remote(...)` call at [`run_online_rl.py:88-93`](../../run_online_rl.py#L88-L93) to include `use_hil_buffer=True`.

## Q15. How do I get the residual to start from random instead of zeros?

It already does. Before the trainer has pushed weights, the control loop emits a uniform-random residual in `[-0.08, 0.08]` ([`control_loop.py:200-201`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L200-L201)):

```python
if not residual_policy_updated:
    residual_action = np.random.uniform(-0.08, 0.08, size=base_policy_action.shape)
```

After the first trainer push, the in-process `residual_policy` is used instead, and that one is Xavier-initialized unless a checkpoint is provided.

If you want a different exploration distribution, this is the place to edit. Be careful: the `0.08` value was chosen empirically — see commit `3fd37e7 increased noise for residual rl`.

## Q16. The action mux uses shortest-arc interpolation for joints. Why?

The igris_b joints are angle-valued and wrap modulo `2π`. A naive linear interpolation between `+170°` and `-170°` would take the long way around (through `0°`), which is mechanically wrong. `IgrisBInterpolator._angle_diff` ([`igris_b_interpolator.py:16-17`](../../env_actor/human_in_the_loop/action_mux/interp_utils/robots/igris_b/igris_b_interpolator.py#L16-L17)) wraps the difference into `[-π, π]` and uses that as the direction. Fingers are normalized `[0, 1]`, so they are linearly interpolated and clipped — no wrapping needed.
