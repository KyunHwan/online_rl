← Back to [docs/residual_rl/README.md](./README.md)

# 05 — Config and hyper-parameters

This is the reference card for every new or changed knob the branch introduces. For the three pre-existing config layers (train YAML / policy YAML / runtime JSON) see [docs/04_configuration.md](../04_configuration.md).

## Table of contents

- [Precedence](#precedence)
- [Entry-point CLI flags](#entry-point-cli-flags)
- [Replay-buffer construction args](#replay-buffer-construction-args)
- [Residual policy YAML](#residual-policy-yaml)
- [Trainer experiment YAML](#trainer-experiment-yaml)
- [DSRL-OpenPI YAML deltas](#dsrl-openpi-yaml-deltas)
- [Hardcoded numbers worth knowing](#hardcoded-numbers-worth-knowing)
- [Tunables, by use case](#tunables-by-use-case)

---

## Precedence

```
CLI flag  >  YAML file  >  code default in run_online_rl.py
```

Inside the trainer, the precedence is:

```
YAML field  >  code default
```

There is no environment-variable layer on this branch. If you set `OMP_NUM_THREADS` or similar you do that in `env_setup.sh` and `start_ray.sh`.

## Entry-point CLI flags

All defined in [`run_online_rl.py:191-228`](../../run_online_rl.py#L191-L228).

| Flag | Type | Default | Action |
|---|---|---|---|
| `--robot` | str | `igris_b` | Selects which `RuntimeParams` to import (`igris_b` or `igris_c`). |
| `--human_reward_labeler` | bool flag | `False` | If set, spawns `ManualRewardLabelerActor` on the labeling PC instead of the auto labeler. |
| `--train_config` | str (path) | `.../trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml` | The training-side YAML. For residual RL, point it at [`.../resfit/online_rl/resfit.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml). |
| `--policy_yaml` | str (path) | `./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml` | The **base** policy YAML. |
| `--residual_policy_yaml` | str (path) | `./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml` | The **residual** policy YAML. The default is wrong for the new flow — pass `./env_actor/policy/policies/resfit_policy/resfit_policy.yaml`. |
| `--use_residual_rl` | bool flag | `False` | Toggle residual mode. When set, the control loop owns the residual policy, the inference loop stops reloading weights, and `ResfitReplayBufferActor` will store `base_policy_action`. |
| `--use_human_intervention` | bool flag | `False` | Selects the HIL `RTCActor` / `SequentialActor` package. Required for teleop pedal + master arm + Manus glove. |
| `--inference_runtime_params_config` | str (path) | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json` | Robot runtime JSON (HZ, action dim, image shape, …). Format documented in [docs/04_configuration.md](../04_configuration.md). |
| `--inference_runtime_topics_config` | str (path) | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json` | Robot ROS topic JSON. |
| `--inference_algorithm` | choice | `rtc` | `rtc` (real-time chunking, the residual-RL path) or `sequential`. |
| `--num_labeler_gpus` | int | `4` | Number of auto-labeler GPU workers to spawn. |

The combination you almost certainly want for the residual-RL smoke test:

```bash
python run_online_rl.py \
  --robot igris_b \
  --train_config <path>/resfit.yaml \
  --policy_yaml ./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml \
  --residual_policy_yaml ./env_actor/policy/policies/resfit_policy/resfit_policy.yaml \
  --use_residual_rl \
  --inference_algorithm rtc
```

Add `--use_human_intervention` if you have the master arm + pedal hooked up. See [07_running_training.md](./07_running_training.md) for the full procedure.

## Replay-buffer construction args

`ResfitReplayBufferActor.__init__` arguments ([`resfit_replay_buffer.py:12-30`](../../data_bridge/resfit_replay_buffer.py#L12-L30)).

| Arg | Type | Default in class | Value used by `run_online_rl.py` | Effect |
|---|---|---|---|---|
| `capacity` | int | `100_000` | (default) | Maximum number of per-step rows stored in the on-disk memmap. |
| `use_hil_buffer` | bool | `False` | (default) | If `True`, splits HIL and auto rollouts into two buffers and samples 50/50. Currently unused. |
| `proprio_key` | str | `"proprio"` | (default) | TensorDict key for proprio history. |
| `reward_key` | str | `"reward"` | (default) | TensorDict key for per-step reward. |
| `action_key` | str | `"action"` | (default) | TensorDict key for executed actions. |
| `image_keys` | tuple | `("head", "left", "right")` | (default) | Image cameras to chunk. |
| `action_horizon` | int | `50` | **`4`** | Number of *future* action steps per sample. Determines `data["action"].shape[1]`. |
| `reward_horizon` | int | `1` | **`3`** | n-step horizon for TD targets. |
| `obs_subsample_step` | int | `3` | **`3`** | Stride for proprio/image offsets — `arange(action_horizon-1, -1, -obs_subsample_step)`. With `action_horizon=4` and step `3`, this yields offsets `[3, 0]` for proprio and images. |
| `strict_length` | bool | `True` | (default) | Drop episodes shorter than the window. |
| `compile` | bool | `False` | (default) | Whether to torch-compile the SliceSampler. |

If you change `action_horizon` or `reward_horizon` here, you must also change them in [`resfit.yaml: data.datamodule.params`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml#L41-L55) so the offline dataset and the online buffer return the same shapes. The trainer concatenates them with `torch.cat(..., dim=0)` ([online_trainer.py:467-504](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L467-L504)).

## Residual policy YAML

### `resfit_policy.yaml`

[`resfit_policy.yaml`](../../env_actor/policy/policies/resfit_policy/resfit_policy.yaml):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model.component_config_paths.resfit_residual_actor` | path | `components/resfit_residual_actor.yaml` | GraphModel component built by the policy factory. |
| `policy.type` | str | `resfit_policy` | Registry key for `ResidualPolicy`. |
| `policy.params.checkpoint_path` | str \| null | `null` | If set, loads `<checkpoint_path>/resfit_residual_actor.pt` at construction time. |
| `policy.params.obs_proprio_history` | int | `1` | Number of proprio timesteps the policy expects — keep at 1 unless retraining. |

### `components/resfit_residual_actor.yaml`

[`resfit_residual_actor.yaml`](../../env_actor/policy/policies/resfit_policy/components/resfit_residual_actor.yaml):

| Field | Default | Effect |
|---|---|---|
| `params.img_resize` | `true` | Resizes images to 240×320 before the ResNet34 group. |
| `params.depth_data_keys` | `[]` | Depth cameras present (none on igris_b). |
| `params.img_data_keys` | `[head, left, right]` | Required camera keys. |
| `params.input_img_channel` | `512` | Channel count of the ResNet34 feature map (block 4). Must match `Resnet34Group`. |
| `params.output_img_channel` | `24` | 1×1 conv output channel for each camera. |
| `params.input_dim` | `5808` | Hardcoded — recompute if camera count or proprio/action width changes. |
| `params.action_dim` | `24` | Residual output dimension. |
| `params.num_layers` | `5` | Hidden layers of the MLP. |
| `params.num_hidden_dim` | `2048` | MLP width. |
| `params.dropout` | `0.0` | No dropout. |
| `params.init_method` | `'xavier'` | Xavier init for MLP + last linear. |

The two YAMLs must agree exactly with the training-side YAMLs at [`experiment_models/resfit/exp1/resfit_residual_actor.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_models/resfit/exp1/resfit_residual_actor.yaml) — otherwise loaded checkpoints will fail.

## Trainer experiment YAML

[`resfit.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml).

| Section | Field | Value | Notes |
|---|---|---|---|
| `plugins` | (list) | dataloader, l2 loss, adamw_cosine, resfit_trainer | Loaded by `load_plugins`. The `l2` loss plugin is loaded but never used by `resfit_trainer`. |
| `model.find_unused_parameters` | bool | `true` | Needed because Q-function ensemble randomly subsamples halves. |
| `model.component_build_args.resfit_residual_actor` | dict | `{init: false, freeze: false, online_update: true}` | `online_update: true` is what makes the trainer push weights to `policy_state_manager`. |
| `model.component_build_args.resfit_q_function` | dict | `{init: false, freeze: false, online_update: false}` | Q-function stays on the training PC; never shipped to inference. |
| `model.component_optims.resfit_residual_actor.params.peak_lr` | float | `1.0e-4` | AdamW warm-cosine peak. |
| `model.component_optims.resfit_q_function.params.peak_lr` | float | `3.0e-4` | Higher than the actor — typical for actor-critic. |
| `model.component_optims.*.total_steps` | int | `200000` | Cosine decay horizon. |
| `model.component_optims.*.warmup_steps` | int | `2000` | Linear warmup before peak. |
| `data.datamodule.params.action_horizon` | int | `4` | Must match buffer. |
| `data.datamodule.params.reward_horizon` | int | `3` | Must match buffer. |
| `data.datamodule.params.obs_proprio_history` | int | `1` | Must match `resfit_policy.yaml`. |
| `data.datamodule.params.HZ` | float | `20` | Used only to convert step offsets into `dt` seconds in `delta_timestamps`. The actual runtime HZ is read from `RuntimeParams`. |
| `data.batch_size` | int | `128` | Both online and offline batches are this size; the trainer concatenates them, so the effective gradient batch is `2 × 128 = 256`. |
| `data.num_workers` | int | `12` | DataLoader workers for the offline LeRobot dataset. |
| `train.save_every` | int | `20` | Checkpoint cadence in iterations (× 25 in `online_trainer.py:526`). |
| `train.save_dir` | path | `/home/user/Projects/online_rl/trainer/experiment_training/reinforcement_learning/resfit/online_rl` | Where epoch checkpoints land. **This is a user-specific absolute path. Change it.** |

Note: `train.epoch: 100000` is a soft target — the loop runs while the offline dataloader has data and the online buffer is alive; the only exit is exception or external `kill`.

## DSRL-OpenPI YAML deltas

These are not residual-RL knobs, but they were changed on this branch to match the checkpoint actually used during training:

| File | Field | `main` | `features/residual_rl` |
|---|---|---|---|
| `noise_actor.yaml` | `params.input_dim` | `49584` | `6960` |
| `noise_processor.yaml` | `params.input_img_channel` | `1024` | `512` |
| `openpi_model.yaml` | `default_prompt` | "Use the left hand to pick up the socks…" | "Use the right hand to pick up the socks…" |

If you swap the base checkpoint, you may need to revisit these. They are decoupled from the residual config.

## Hardcoded numbers worth knowing

These are constants in code, not config. Search-and-replace if you need to change them.

| Value | Location | What it controls |
|---|---|---|
| `1000` | [`control_loop.py:91`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L91) | Max control steps per episode (≈ 25 s at 40 Hz). |
| `35` | [`inference_loop.py:18`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L18) | `min_num_actions_executed` — the threshold the inference loop waits for before generating the next chunk. |
| `np.random.uniform(-0.08, 0.08, ...)` | [`control_loop.py:201`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L201) | Bootstrap exploration residual before the first trainer push. |
| `0.25` | [trainer:residual_actor.py:222](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py#L222) | Residual output ceiling (Tanh scale). |
| `20` | [`action_mux.py:31`](../../env_actor/human_in_the_loop/action_mux/action_mux.py#L31) | Number of interpolation steps for POLICY→TELEOP blending. |
| `10` | [`arms_dynamixel.py:181`](../../env_actor/human_in_the_loop/teleoperation/robots/igris_b/arms_dynamixel.py#L181) | Consecutive sync-read failures before raising. |
| `10` | [trainer:residual_q_function.py:185](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_q_function.py#L185) | Q-function ensemble size. |
| `0.005` | [trainer:critic_trainer.py:13](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L13) | Polyak target-update rate τ. |
| `0.99` | [trainer:critic_trainer.py:14](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L14) | Discount factor γ. |
| `% 10 == 0` | [trainer:resfit_trainer.py:92](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py#L92) | Actor-update cadence (one actor step per 10 critic steps). |
| `25 × save_every` | [trainer:online_trainer.py:526](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L526) | Trainer-side checkpoint + weight-push cadence. |

## Tunables, by use case

| You want to... | Touch this knob |
|---|---|
| Reduce GPU memory during training | `data.batch_size`, `data.num_workers` |
| Train longer | `train.epoch`, `model.component_optims.*.total_steps` (do both) |
| Push weights more often | `train.save_every` (lower) |
| Sharpen the residual ceiling | `0.25` in [trainer:residual_actor.py:222](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py#L222) **and** retrain — do not change at inference only |
| Reduce exploration before the first push | The `0.08` in [`control_loop.py:201`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L201) |
| Smooth pedal transitions more | `interp_steps` in [`action_mux.py:31`](../../env_actor/human_in_the_loop/action_mux/action_mux.py#L31) |
| Use a different LeRobot dataset | `data.datamodule.params.repo_id` in `resfit.yaml` |
| Change replay-buffer horizons | Update **both** `resfit_replay_buffer` defaults (in [`run_online_rl.py:88-93`](../../run_online_rl.py#L88-L93)) **and** `resfit.yaml: data.datamodule.params` |
| Run on CPU only | Not supported. The ResNet34 + bf16 autocast path requires CUDA; see [10_faq_onboarding.md](./10_faq_onboarding.md) |
