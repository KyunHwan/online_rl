← Back to [docs/residual_rl/README.md](./README.md)

# 09 — Glossary

Alphabetical. Only terms that show up in the codebase or in these residual-RL docs.

| Term | Definition |
|---|---|
| **Action chunk** | A sequence of actions emitted by a single forward pass of the base policy. For igris_b, `action_chunk_size = 50` at the runtime layer (see [`runtime_settings_configs/`](../../env_actor/runtime_settings_configs/)) and `action_horizon = 4` at the replay-buffer layer. |
| **Actor (RL)** | The policy network. On this branch the trainable actor is the **residual actor** `Residual_Actor` (see [`residual_actor.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/policy_constructor/model_constructor/blocks/experiments/resfit/residual_actor.py)). |
| **Actor-critic** | An RL family where an actor outputs actions and a critic evaluates them. `resfit_trainer` is actor-critic. |
| **Anchor (replay buffer)** | The timestep within a chunked sample treated as "now." Defined by `-min(offset)` in [`_compute_window`](../../data_bridge/resfit_replay_buffer.py#L99-L105). For `action_horizon=4, reward_horizon=3, obs_subsample_step=3` the anchor is index 0. |
| **AutoCast (bf16)** | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — mixed-precision inference. Used both in the inference loop and in the trainer to halve memory and double-ish throughput. |
| **Base policy** | The fixed policy whose action chunks are corrected by the residual. On this branch it is the DSRL-OpenPI VLA. |
| **`base_policy_action`** | The action chunk produced by the base policy. Stored in both the runtime episode recorder and the replay buffer alongside the executed action. |
| **Behavioral cloning (BC)** | Supervised learning of an action from expert states. *Not* what this branch optimizes — the residual is trained against a learned Q-function. |
| **Control loop** | The 40 Hz process that reads sensors, decides on the next action, and publishes it to the robot. On this branch it owns the residual policy. |
| **Critic (RL)** | The Q-function. On this branch it is the 10-MLP `Q_Function` ensemble. |
| **DDP** | `torch.nn.parallel.DistributedDataParallel`. The trainer wraps policies in DDP automatically through `ray.train.torch.prepare_model`. |
| **`delta_timestamps`** | LeRobot's mechanism for asking the dataloader to return future / past frames relative to a sample. Used by the offline trainer to mirror the online buffer's chunked output. |
| **DSRL-OpenPI** | The base policy architecture used here. A diffusion-style VLA built on the [openpi](https://github.com/Physical-Intelligence/openpi) backbone. See [`policy/policies/dsrl_openpi_policy/`](../../env_actor/policy/policies/dsrl_openpi_policy/). |
| **Episode** | A sequence of `episode_length` control steps (currently 1000) terminated by a stop event or step count. Sub-divided into sub-episodes for labeling and storage. |
| **Episode recorder** | The actor that accumulates per-step observations and actions in memory and serves them as TensorDict lists at episode boundaries ([`episode_recorder_bridge.py`](../../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py)). |
| **Frozen policy** | A policy whose weights are loaded from a checkpoint and not updated at runtime. The base policy is frozen when `use_residual_rl=True`. |
| **GAE (Generalized Advantage Estimation)** | Standard RL trick. **Not** used on this branch — the critic uses an n-step TD target. Listed for cross-reference only. |
| **Git submodule** | A repo embedded inside another repo. The `trainer/` directory is a submodule of `online_rl/`. See [docs/10_glossary.md#git-submodule](../10_glossary.md#git-submodule). |
| **Guided inference** | Action-chunk blending technique that weights newly-predicted actions against the previously-emitted chunk, using `compute_guided_prefix_weights` from [`action_inpainting.py`](../../env_actor/inference_engine_utils/action_inpainting.py). |
| **HIL (Human-in-the-loop)** | The teleop intervention path: a pedal switches between policy and teleop output; teleop action comes from the Dynamixel master arm + Manus glove. Lives under [`env_actor/human_in_the_loop/`](../../env_actor/human_in_the_loop/). |
| **igris_b / igris_c** | The two robot families this codebase supports. igris_b is the active one on this branch. |
| **Imitation learning** | Supervised learning from demonstrations. The previous training mode (before residual RL) was an imitation-learning fine-tune. |
| **Inference loop** | The process that runs the base policy's expensive forward pass, asynchronously to the control loop. Writes action chunks to shared memory; the control loop consumes them at 40 Hz. |
| **`init_action_chunk_obs_history`** | New shared-memory bridge method that seeds the proprio history with the current state and the action chunk with the robot's home pose, at every episode start ([`shm_manager_bridge.py:367-394`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py#L367-L394)). |
| **KL penalty** | Standard PPO-style regularizer. **Not** used here. |
| **`LazyMemmapStorage`** | TorchRL on-disk storage for the replay buffer. The buffer wipes `/tmp/online_rl_auto_data/` at startup to avoid stale-file conflicts. |
| **LeRobot** | Hugging Face robotics dataset library. The offline data source ([`resfit_lerobot_data.py`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/dataloader/resfit_lerobot_data.py)) reads `joon001001/igris-b-pnp-v4`. |
| **`min_num_actions_executed`** | Threshold (35) that the inference loop waits for before generating the next chunk. Acts as a "the control loop has consumed enough of the previous chunk that we should plan again" signal. |
| **n-step TD target** | The critic loss target is `R_chunk + γ^H · Q_target(s_{t+H}, a_{t+H})`, where `R_chunk` is a discounted sum over `reward_horizon` rewards and `H = reward_horizon`. See [`critic_trainer.py:60-71`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/components/trainer/reinforcement_learning/resfit/utils/critic_trainer.py#L60-L71). |
| **Off-policy** | Training a policy on data that was generated by a *different* (typically older) policy. The replay buffer makes this branch off-policy. |
| **On-policy** | Training on data generated by the current policy. Not how the residual is trained, though the actor loss is "on-policy" in the weak sense that it uses the *current* residual when evaluating the Q. |
| **`online_update`** | Per-model flag in the trainer YAML. If `true`, the trainer pushes that model's CPU state dict to `policy_state_manager` after each checkpoint. Only `resfit_residual_actor` is `online_update: true`. |
| **OpenPI** | The diffusion-VLA backbone the base policy uses. The local checkpoint patcher is [`openpi_transformer_lib_patch.sh`](../../openpi_transformer_lib_patch.sh). |
| **`policy_state_manager`** | The named Ray actor that holds the latest pushed model weights for the inference side ([`data_bridge/state_manager.py`](../../data_bridge/state_manager.py)). |
| **Polyak averaging** | Slow-tracking exponential update of a target Q-network: `θ_target ← (1-τ)·θ_target + τ·θ_source`. `τ = 0.005` here. |
| **Q-ensemble** | A collection of independent Q-networks. Their mean is used as the value estimate; a random half is used for the critic loss to reduce overestimation bias. |
| **Q-function `Q(s, a)`** | Estimates the expected return of taking `a` in `s` then following the policy. Trained against an n-step TD target. |
| **Ray** | Distributed-task framework. Two notions matter here: **Ray actors** (long-lived stateful workers identified by name and Ray namespace) and **Ray remotes** (one-shot calls returning ObjectRefs). |
| **Ray actor** | A long-lived Python class running on a Ray worker. `RTCActor`, `ResfitReplayBufferActor`, `StateManagerActor`, the reward labelers — all Ray actors. |
| **`RAYQUEUE_MAXSIZE`** | Capacity of the `RayQueue` between the control loop and the reward labelers. Currently 5 ([`run_online_rl.py:33`](../../run_online_rl.py#L33)). |
| **Replay buffer** | The persistent record of past transitions. Off-policy methods sample from it to train. Here: `ResfitReplayBufferActor`. |
| **Residual policy** | A policy that produces a *delta* on top of the base action: `a_final = base + residual`. |
| **Residual RL** | RL where the policy is decomposed into a fixed base + a trainable residual. This branch's whole point. |
| **`Resnet34Group`** | A `nn.ModuleDict` of three pretrained ResNet-34 backbones (head, left, right), shared by both `Residual_Actor` and `Q_Function`. |
| **Return** | Sum of discounted future rewards. The Q-function approximates the return. |
| **Reward labeler** | An actor that pops sub-episodes from the queue and computes a per-step reward. Auto (`AutoRewardLabelerActor`) and manual (`ManualRewardLabelerActor`) variants exist. |
| **Rollout** | A run of the policy in the environment producing trajectories. The control-loop process does the rolling out. |
| **`RTCActor`** | The Ray remote that orchestrates the RTC inference algorithm. Spawns two child processes (control + inference). |
| **RTC (Real-Time Chunking)** | The chunked-action inference algorithm: the policy emits a chunk, the control loop replays the chunk while the policy plans the next one in parallel. |
| **`shared_memory` (multiprocessing)** | OS-backed shared memory used to pass action chunks and observation buffers between the inference and control processes. |
| **`SliceSampler`** | TorchRL replay-buffer sampler that returns contiguous slices of length `slice_len`, one per requested sample. |
| **State dict** | `torch` term for a flat ordered dict of parameter tensors. Used as the wire format for the weight-push channel. |
| **Sub-episode** | A slice of an episode produced by the episode recorder for labeling. One episode → one or more sub-episodes. |
| **τ (tau, Polyak rate)** | `target_update_rate = 0.005`. Lower means slower target tracking. |
| **TD3 / Twin critics** | The TD3 algorithm uses two critics and takes the min to mitigate overestimation. This branch generalizes to 10 critics + random half-sampling. |
| **TensorDict** | The `torchrl/tensordict` data structure: a nested dict of tensors with a shared batch shape. The episode recorder and replay buffer both use it. |
| **VLA (Vision-Language-Action)** | A class of policies that condition on images + text and emit actions. DSRL-OpenPI is a VLA. |
| **Wandb** | Weights & Biases experiment tracker. Trainer rank 0 writes to project `resfit_online_rl`. |
| **Xavier init** | Glorot uniform initialization. Used for the MLP and last linear layers of the residual actor and the Q-function ensemble. |
| **γ (gamma)** | Discount factor for future rewards. `0.99` here. |
