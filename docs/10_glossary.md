# 10 — Glossary

Definitions of terms used across the docs. One or two sentences each, with links to upstream when there is one.

## Action chunk

A batch of N future actions predicted in a single policy call, instead of one action at a time. The policy produces `(action_chunk_size, action_dim)` per inference; the control loop executes them in sequence until the next inference call lands.

## Action chunking

The technique of predicting many future actions at once and executing them over multiple control steps. Reduces the number of forward passes per second the policy has to make, which makes large vision-language-action models tractable on real-time control loops.

## Action horizon

How many future actions a chunk covers. In this repo, `action_horizon` (used by the replay buffer) and `action_chunk_size` (used by the runtime) refer to the same number — 50 by default. The replay buffer's `_build_offsets()` exposes `action_horizon`; the runtime JSON calls it `action_chunk_size`.

## Action inpainting

The trick that smooths over the gap between the action chunk a policy predicted N control-steps ago and the chunk it just produced. Weights ≈ 1 over the prefix that the previous chunk has already committed to, weights → 0 over the tail where the new prediction takes over. Implemented by [`compute_guided_prefix_weights()`](../env_actor/inference_engine_utils/action_inpainting.py) in this repo. See [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339).

## Autocast

`torch.autocast` is the context manager that automatically casts ops to a lower precision (here, `bfloat16`) for speed while keeping certain ops in float32 for stability. The RTC inference loop wraps `policy.guided_inference()` in `with torch.autocast(device_type="cuda", dtype=torch.bfloat16):`. Trainer-side details in [trainer/docs/04_concepts.md](../trainer/docs/04_concepts.md).

## Custom resource

A label like `"training_pc"` or `"inference_pc"` declared via `ray start --resources='{"<label>": N}'`. Actors with `.options(resources={"<label>": K})` only land on nodes that declared at least `K` units of that label. Used in this repo to pin actors to specific machines. See [03_distributed_setup.md](03_distributed_setup.md#how-resources-pin-actors-to-machines).

## DDP

Distributed Data Parallel — PyTorch's standard multi-GPU training mechanism. Each worker holds a full model replica, computes gradients on its own minibatch, and `AllReduce`s them. The trainer half of this repo runs DDP with `num_workers=4` under Ray Train; see [trainer/docs/02_distributed_training.md](../trainer/docs/02_distributed_training.md).

## DSRL

The "Diffusion Steering for Reinforcement Learning" recipe wired into the default policy of this repo (`dsrl_openpi_policy`). A DSRL actor produces a structured noise tensor; an OpenPI diffusion model then converts that noise into actions. The combination lets gradient signals from RL flow through the noise, while keeping the expressive diffusion policy frozen-or-slow-updating.

## Episode queue

The `ray.util.queue.Queue` (`maxsize=15`) created in [run_online_rl.py](../run_online_rl.py) that carries completed episodes (as `ray.put(TensorDict)` refs) from the env actor to the reward labeler. Bounded — when full, the env actor blocks on `put`, which throttles episode production to labeler speed.

## Flow matching

A class of generative models (close cousin of diffusion) that learns a continuous-time velocity field which transports samples from a simple noise distribution to the data distribution along ODE trajectories. OpenPI is a flow-matching policy: at inference, an Euler ODE solve converts noise → action chunk. The math reference: [Lipman et al. (2023), "Flow Matching for Generative Modeling"](https://arxiv.org/abs/2210.02747).

## GIL

The CPython Global Interpreter Lock — one thread executes Python bytecode at a time per interpreter. The reason RTC uses two **processes** (not threads): GPU inference in one thread would block the realtime control loop in another. See [02_architecture.md](02_architecture.md#the-rtc-two-process-model).

## Git submodule

A Git repository nested inside another. This repo uses two: [trainer/](../trainer/) and [data_labeler/auto/models/robometer/](../data_labeler/auto/models/robometer/). After `git clone`, you must `git submodule update --init --recursive` to populate them.

## Guided inference

The general term for inference that conditions on prior context (the previous action chunk's tail, an estimated delay). In this codebase it refers specifically to `policy.guided_inference(...)` — the RTC inference path's entry into the policy. The "guidance" is action inpainting; the function name is a historical artifact.

## HIL

Human-in-the-loop. Refers to operator intervention during a policy run — a foot pedal swaps control from the policy to a teleoperator. Scaffolded in [env_actor/human_in_the_loop/](../env_actor/human_in_the_loop/) but not wired into the live entrypoint today.

## LeRobot

[Hugging Face's robotics dataset and training library](https://github.com/huggingface/lerobot). This repo's replay buffer mimics LeRobot's chunked-sample naming (`observation.state`, `observation.images.cam_*`, `labels.reward`, etc.) so trainer code can be shared between offline LeRobot-format data and the online replay buffer.

## Memmap

`numpy.memmap` / torchrl's `LazyMemmapStorage` — maps a file on disk into memory so reads and writes look like regular array accesses but are backed by disk. The replay buffer uses this to avoid keeping millions of timesteps in RAM; the OS pages data in and out on demand.

## Normalization stats

A pickle file with `mean` and `std` arrays for proprio, current, and action — produced during dataset preparation, consumed at inference time by [DataNormalizationBridge](../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py). Path is configured in the runtime JSON's `norm_stats_file_path` key.

## OpenPI

[Physical Intelligence's open-source VLA policy](https://github.com/Physical-Intelligence/openpi), based on a Gemma-3 / SigLIP vision-language backbone with a flow-matching action head. The PyTorch port ships under [trainer/policy_constructor/.../third_party/openpi/](../trainer/policy_constructor/) (inside the trainer submodule). Used directly via `openpi_policy` and as one component of `dsrl_openpi_policy`.

## Plasma object store

Ray's in-memory shared-object store. `ray.put(obj)` writes `obj` into Plasma and returns a small `ObjectRef`; any actor on the same node reads `obj` zero-copy. Used in two places here: (1) episode TensorDicts go in via `ray.put` before the queue, (2) state-dict refs go in before `StateManager.update_state(ref)`.

## Policy update period

The number of control steps between policy invocations in the **Sequential** algorithm. With `policy_update_period=50` and `HZ=20`, the policy runs once every 2.5 seconds, executing the chunked actions in between. Unused by RTC (RTC's pacing is driven by `min_num_actions_executed` and the shared-memory condition variables).

## Proprioception

Sensor readings about the robot's own state: joint positions, joint currents, finger angles. The `proprio` numpy array is the flat concatenation of all selected per-link readings, listed in `IGRIS_B_STATE_KEYS` ([env_actor/runtime_settings_configs/robots/igris_b/init_params.py](../env_actor/runtime_settings_configs/robots/igris_b/init_params.py)).

## Ray

[Ray](https://docs.ray.io/) is a distributed-Python runtime. This repo uses it for three things: actors (long-lived Python objects with their own process), tasks (one-shot remote functions), and Ray Queue (a distributed FIFO). The trainer additionally uses Ray Train for DDP orchestration.

## Ray actor

A long-lived stateful Python object running in its own Ray worker process. Decorated with `@ray.remote`. Constructed via `MyActor.remote(...)`; methods called via `actor.method.remote(args)` return `ObjectRef`s. Pinned to a machine via `.options(resources=...)`. See [02_architecture.md](02_architecture.md) for the four actor types used here.

## Replay buffer

A storage actor that holds collected experience and serves randomly-sampled training batches. This repo's is [`ReplayBufferActor`](../data_bridge/replay_buffer.py) — a Ray actor wrapping `TensorDictReplayBuffer` from torchrl, backed by `LazyMemmapStorage` on disk.

## Reward labeler

The component that decides what the per-frame reward is for a recorded episode. Two flavors:
- **Auto** ([Robometer](#robometer) VLM) — scores each frame's task progress and uses that as reward.
- **Manual** (PySide6 GUI) — a human scrubs through frames and assigns binary rewards.

## Robometer

A vision-language reward model trained to predict task progress and success probability from robot video. Lives as a submodule under [data_labeler/auto/models/robometer/](../data_labeler/auto/models/robometer/). Loaded as `robometer/Robometer-4B` from HuggingFace by the auto labeler.

## RTC

Real-Time action Chunking. The default inference algorithm in this repo. Splits inference and control into two OS-level processes communicating through shared memory, so the realtime control loop is not blocked by GPU inference. See [02_architecture.md](02_architecture.md#the-rtc-two-process-model).

## SharedMemory

`multiprocessing.shared_memory.SharedMemory` — a POSIX shared-memory block accessible across multiple processes by name. RTC allocates 5 of these (proprio, three cameras, action chunk) and both the inference and control processes attach to them. See [shared_memory_utils.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py).

## SliceSampler

torchrl's sampler that draws fixed-length trajectory slices from a replay buffer (rather than i.i.d. timesteps). [docs](https://pytorch.org/rl/stable/reference/data.html#slicesampler). Used here to produce windows long enough to cover the full proprio-history + action-horizon offset range. The window length comes from `_build_offsets` and `_compute_window` in [replay_buffer.py](../data_bridge/replay_buffer.py).

## Slew-rate limiting

Capping per-step joint motion to `max_delta` radians. Implemented in [ControllerBridge.publish_action()](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py): `delta = np.clip(raw_joint - prev_joint, -max_delta, max_delta); smoothed_joints = prev_joint + delta`. Protects the robot from sudden large commands; `max_delta_deg=5` in the default JSON, i.e. ~8.7e-2 rad per 50 ms step.

## Spawn (multiprocessing context)

`multiprocessing.get_context("spawn")` — starts new processes by forking and `exec`ing Python from scratch, rather than `fork`ing the current interpreter. Required when you have CUDA initialized in the parent: `fork` after CUDA init leaves the child with broken CUDA contexts.

## State manager

The Ray actor that holds the latest policy-weight reference and the version counters. Trainer pushes; inference loop pulls; counters gate redundant transfers. See [data_bridge/README.md](../data_bridge/README.md#statemanageractor).

## Tailscale

A managed WireGuard-based overlay VPN. Gives every machine a stable IP in `100.x.y.z` regardless of the underlying network. Used in [start_ray.sh](../start_ray.sh) to let three machines find each other across NATs and subnets. [tailscale.com](https://tailscale.com/).

## TensorDict

A dict-like container of tensors with a shared batch shape. [Docs](https://pytorch.org/tensordict/stable/index.html). Lets you slice, stack, and reshape a nested set of tensors as one object — `td[:, 5:10]` slices every tensor inside on its time dimension. This repo passes episodes and replay-buffer samples as TensorDicts.

## torchrl

PyTorch's RL library. [Docs](https://pytorch.org/rl/stable/). This repo uses `TensorDictReplayBuffer`, `LazyMemmapStorage`, and `SliceSampler` from it.

## VLM

Vision-Language Model. A neural network that ingests both images and text. Robometer is a VLM (it reads task descriptions and video frames, outputs progress scores). OpenPI is a Vision-Language-**Action** Model, a VLM whose output head produces actions instead of text.
