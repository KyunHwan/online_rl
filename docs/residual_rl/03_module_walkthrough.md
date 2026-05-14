← Back to [docs/residual_rl/README.md](./README.md)

# 03 — Module walkthrough

This is a directory-by-directory tour of every file the branch touches in the **parent repo**. The submodule walkthrough is in [04_trainer_submodule_changes.md](./04_trainer_submodule_changes.md).

Every entry below uses the same three-part shape:

- **Change** — what was added, modified, or deleted (with code citations).
- **Why** — the rationale a junior engineer needs to internalize before they next have to touch this code.
- Code excerpts where the decisive logic fits in a few lines.

## Table of contents

- [data_bridge/](#data_bridge)
- [env_actor/auto/inference_algorithms/rtc/](#env_actorautoinference_algorithmsrtc)
- [env_actor/episode_recorder/](#env_actorepisode_recorder)
- [env_actor/human_in_the_loop/action_mux/](#env_actorhuman_in_the_loopaction_mux)
- [env_actor/human_in_the_loop/inference_algorithms/rtc/](#env_actorhuman_in_the_loopinference_algorithmsrtc)
- [env_actor/human_in_the_loop/inference_algorithms/sequential/](#env_actorhuman_in_the_loopinference_algorithmssequential)
- [env_actor/human_in_the_loop/intervention_methods/pedal/](#env_actorhuman_in_the_loopintervention_methodspedal)
- [env_actor/human_in_the_loop/io_interface/ (deleted)](#env_actorhuman_in_the_loopio_interface-deleted)
- [env_actor/human_in_the_loop/teleoperation/](#env_actorhuman_in_the_loopteleoperation)
- [env_actor/policy/policies/](#env_actorpolicypolicies)
- [env_actor/robot_io_interface/](#env_actorrobot_io_interface)
- [Root-level files](#root-level-files)

---

## `data_bridge/`

Holds the data-side actors that bridge episode rollouts to the trainer.

### `data_bridge/replay_buffer.py` — modified

**Change.** The pre-existing imitation-learning buffer was kept and lightly modified:

- Capacity dropped `10_000_000` → `100_000`.
- `compile=True` → `compile=False` on `SliceSampler`.
- Scratch dirs moved from relative `tmp/online_rl_*_data` to absolute `/tmp/online_rl_*_data`; the buffer now also wipes the directory at startup ([`replay_buffer.py:54-58`](../../data_bridge/replay_buffer.py#L54-L58)).
- Offset construction was reworked so the `classic` mode now indexes proprio backwards in time from "now," mirroring the runtime ring buffers ([`replay_buffer.py:86-115`](../../data_bridge/replay_buffer.py#L86-L115)).
- `_pack_lerobot_like` now emits `base_policy_action` when present ([`replay_buffer.py:179-184`](../../data_bridge/replay_buffer.py#L179-L184)).

**Why.** This file is the *legacy* imitation-learning buffer; residual RL uses `resfit_replay_buffer.py` (next section). The capacity drop and absolute-path move fix two specific production problems: (1) the 10 M-row memmap exhausted the `/tmp` partition on the training PC, (2) relative scratch paths created the memmap next to whichever working directory Ray happened to set, leading to stale-file races. `compile=False` is from commit `fc177a7`: the compiled sampler was unstable when the memmap was resized between runs. The proprio offset reorder makes the *offline* buffer align with how the *online* buffer reads `obs` (newest at index 0) so any callers that swap one for the other do not silently invert their history. The `base_policy_action` pack lets this legacy buffer also be used by a residual training run in a pinch.

### `data_bridge/resfit_replay_buffer.py` — new

**Change.** New `ResfitReplayBufferActor` class ([`resfit_replay_buffer.py:11`](../../data_bridge/resfit_replay_buffer.py#L11)). Three integer knobs (`action_horizon=4`, `reward_horizon=3`, `obs_subsample_step=3`) drive offset construction in [`_build_offsets`](../../data_bridge/resfit_replay_buffer.py#L83-L97). The buffer stores per-step rows including `base_policy_action` and emits **chunked windows** through `_pack_lerobot_like` with LeRobot key names (`observation.images.cam_head`, `labels.reward`, etc.).

**Why.** The residual recipe demands three things the legacy buffer cannot provide cleanly:

1. **Different horizons for action and reward.** The actor consumes a 4-step action chunk; the critic uses 3-step n-step TD targets. Forcing both into a single horizon throws away samples or wastes compute.
2. **Aligned `base_policy_action`.** At training time the actor loss is `-Q(s, a_base + residual(s, a_base)).mean()`. Without persisting the base action that was *actually* executed at rollout time, the critic and actor would disagree about what "the base policy did" — they would each rerun the base policy on stored observations, double the inference cost on the training PC, and possibly differ from rollout-time behavior if the base policy is non-deterministic.
3. **LeRobot-naming output.** The trainer mixes online and offline batches via `torch.cat`; identical keys mean no rename layer is needed. See [04 § integration contract](./04_trainer_submodule_changes.md#integration-contract-with-the-parent-repo).

The HIL split-buffer feature (`use_hil_buffer=True`) is implemented but unused — see [10 Q14](./10_faq_onboarding.md#q14-the-buffers-add-has-a-separate_key-argument--whats-it-for) for when you would turn it on.

A junior should understand three methods only: `add`, `sample`, `size`. The rest is offset bookkeeping that must stay locked to the offline dataloader's `delta_timestamps` in the trainer submodule.

## `env_actor/auto/inference_algorithms/rtc/`

The auto-control RTC tree on `main` already had a clean inference/control split. This branch threads the residual flag through it.

### `rtc_actor.py` — modified

**Change.** Two new constructor args, `residual_policy_yaml_path` and `use_residual_rl`, forwarded into both child processes ([`rtc_actor.py:14-30`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L14-L30), [`:117`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L117), [`:140-141`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L140-L141)).

**Why.** Residual mode is a *runtime* decision (one driver invocation may use it, another may not), so the flag has to flow from the CLI through `run_online_rl.py` through the Ray actor into the spawned processes. Threading it as constructor arguments instead of as a global lets the auto and HIL actors share the exact same spawn machinery — the only difference between modes is which branch the loops take.

### `actors/control_loop.py` — modified

**Change.** The heart of residual inference. The decisive lines:

```python
# env_actor/auto/inference_algorithms/rtc/actors/control_loop.py:195-204
base_policy_action = None
if use_residual_rl:
    base_policy_action = action.copy()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if not residual_policy_updated:
            residual_action = np.random.uniform(-0.08, 0.08, size=base_policy_action.shape)
        else:
            residual_action = residual_policy.inference(base_policy_action, obs_data)
        action = residual_action + base_policy_action
```

Sub-changes:

- **Residual policy construction at startup** ([`control_loop.py:95-105`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L95-L105)).
- **Weight reload polled per episode** ([`control_loop.py:115-127`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L115-L127)).
- **Bootstrap exploration** with `np.random.uniform(-0.08, 0.08)` until the trainer pushes the first weights.
- **Episode-boundary refactor** ([`control_loop.py:163-174`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L163-L174)): the previous separate `reset() / init_action_chunk() / bootstrap_obs_history(...)` is replaced by `reset(); sleep(1.5); init_action_chunk_obs_history(...)`.
- **Episode-queue back-pressure** ([`control_loop.py:151-155`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L151-L155)): `block=True, timeout=30` with a warning on drop.
- **Recorder receives both `action` and `base_policy_action`** ([`control_loop.py:214`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L214)).

**Why each sub-change.**

- **Where to apply the residual.** The residual must run after the inference loop has written the base chunk into shared memory and after the control loop has picked the action of *this* tick from that chunk. Doing the addition in the control loop (not in the policy class) keeps the residual completely outside the inference loop's diffusion-style forward pass, which would otherwise have to be modified to know about a second model.
- **Weight reload per episode.** Reloading mid-episode would change the policy mid-trajectory, breaking the assumption that one rollout was produced by one policy — which the off-policy critic depends on when computing TD targets. Per-episode reload is the coarsest cadence that still keeps the policy fresh.
- **Bootstrap noise.** Until the trainer has produced a first weight push, the in-process residual is Xavier-random. Forwarding it produces structured but useless garbage; injecting uniform noise instead guarantees the buffer fills with exploratory data that the critic can learn from immediately. The `0.08` magnitude was tuned in commit `3fd37e7 increased noise for residual rl` so the bot still completes pick-and-place under random perturbation.
- **Episode-boundary refactor.** Guided inference blends the *new* chunk against whatever is in shared memory. On `main`, that "whatever" was zeros after `init_action_chunk()`, which made the first emitted chunk biased toward the origin pose for one chunk. Seeding the action chunk with the robot's home pose (via the new bridge method) means the first guided-inference output is a sane diff against where the robot actually is. The 1.5 s sleep after `reset()` lets the prior episode's shared-memory traffic drain before the new episode writes.
- **Episode-queue back-pressure.** Previously `put(..., block=True)` could block forever if the labeler crashed, locking the inference machine into an undebuggable hang. The 30-second timeout + drop-with-warning gives a visible signal in the driver log and keeps rollouts flowing rather than deadlocking the whole cluster.
- **Recorder gets both actions.** The replay buffer cannot reconstruct `base_policy_action` after the fact — by the time the trainer reads a sample, both the base policy weights *and* the random state of the chunked inference have moved on. Persisting the rollout-time base action is the only way the critic and actor can share a common reference.

### `actors/inference_loop.py` — modified

**Change.** Weight reloads are gated on `use_residual_rl` ([`inference_loop.py:99-109`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L99-L109)). In residual mode the base policy is loaded once at startup from `policy_yaml_path` and never refreshed; in non-residual mode the previous behavior (poll `policy_state_manager`) is preserved. A `use_residual_rl` flag was added to the function signature ([`inference_loop.py:15`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L15)).

**Why.** Residual RL's central invariant is *the base policy is frozen*. If the inference loop kept polling `policy_state_manager`, the base would get whatever the trainer last pushed — and the trainer pushes the residual actor, not the base. The two would compete for the same shared-state slot. Gating the reload on `not use_residual_rl` is the cheapest way to enforce the invariant without forking the inference loop into two files.

### `data_manager/robots/igris_b/shm_manager_bridge.py` — modified

**Change.** New method `init_action_chunk_obs_history(obs_history)` ([`shm_manager_bridge.py:367-394`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py#L367-L394)). It fills the proprio history with the current state repeated `proprio_history_size` times, copies the latest image frames, and seeds the action buffer with the home pose `INIT_JOINT_LIST` + `INIT_HAND_LIST` (joints to radians, fingers scaled by `0.03`).

**Why.** See the "episode-boundary refactor" rationale above. The two prior calls (`init_action_chunk()` + `bootstrap_obs_history(...)`) had to be invoked in a specific order under the bridge's internal lock; bundling them into one method removes a sharp edge where a tired engineer could call them out of order and produce mismatched action/obs shared memory.

### `data_manager/shm_manager_interface.py` — modified

**Change.** Adds `init_action_chunk_obs_history` to the interface, proxying through to the bridge ([`shm_manager_interface.py:124-127`](../../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py#L124-L127)).

**Why.** The control loop talks to the bridge through the interface; without the proxy method, the loop would have to reach into `self.shm_manager.shm_manager`, which breaks the encapsulation that lets the interface swap bridges between robots.

## `env_actor/episode_recorder/`

### `robots/igris_b/episode_recorder_bridge.py` — modified

**Change.** Three changes:

1. **`add_action` accepts `base_policy_action`** ([`episode_recorder_bridge.py:130-145`](../../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py#L130-L145)).
2. **A `task` field** is written using `NonTensorData`, defaulting to `"pick and place"` if no prompt is set ([`episode_recorder_bridge.py:83-91`](../../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py#L83-L91)).
3. **`init_train_data_buffer(prompt)`** persists the prompt for the next serve call.

**Why.**

1. The residual recipe trains against `data["base_policy_action"]`. The recorder is the only point in the system that knows both numbers at the same instant. If the recorder dropped the base action, the buffer would have no way to reconstruct it.
2. The trainer's offline LeRobot dataset emits a `task` field per sample. Matching it on the online side means the dataloader and the replay buffer return TensorDicts with the same key set, so `torch.cat` on `dim=0` ([online_trainer.py:467-504](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/online_trainer.py#L467-L504)) does not raise on mismatched columns. `NonTensorData` is used because `task` is a string, not a tensor.
3. The prompt is decided once per episode (in the control loop) but consumed at episode-end (when the train-data buffer is served). Keeping it on the recorder avoids passing it through every per-step call.

## `env_actor/human_in_the_loop/action_mux/`

This package selects between policy and teleop actions at every control tick when `--use_human_intervention` is set.

### `action_mux.py` — modified

**Change.** `ActionMux.select` no longer hands the raw teleop action back the instant the pedal switches mode. Instead it:

1. Detects the POLICY → TELEOP transition.
2. Asks a robot-specific `ActionInterpolator` for a `20`-step trajectory from the *last emitted* action to the new teleop target.
3. Emits one trajectory step per `select()` call.
4. On TELEOP → POLICY transitions, no blending; the policy chunk is sent immediately ([`action_mux.py:79-90`](../../env_actor/human_in_the_loop/action_mux/action_mux.py#L79-L90)).

The interpolator is robot-pluggable via `_build_interpolator` ([`action_mux.py:16-20`](../../env_actor/human_in_the_loop/action_mux/action_mux.py#L16-L20)).

**Why.** On `main`, the very first teleop tick after a pedal press emitted the operator's raw joint pose. If the operator was already mid-grasp (joints far from the robot's last commanded pose), the master-arm pose could differ from the robot's pose by tens of degrees, causing a hard joint snap and a safety-stop. The 20-step interpolation gives the operator about 0.5 s at 40 Hz to be at the right pose before the robot is fully there; that is short enough not to feel laggy and long enough to avoid the snap. Only POLICY → TELEOP is blended because the reverse direction (a policy chunk built from the *current* sensor reading) is already smoothly continuous with the robot's pose.

### `interp_utils/__init__.py` — new

**Change.** Package marker file.

**Why.** Standard Python package layout — `from .interp_utils.interp_interface import ActionInterpolator` only works if `interp_utils` is a package.

### `interp_utils/interp_interface.py` — new

**Change.** Abstract base class `ActionInterpolator` ([`interp_interface.py:12-35`](../../env_actor/human_in_the_loop/action_mux/interp_utils/interp_interface.py#L12-L35)). Subclasses must return a list of intermediate `np.ndarray`s from `action_from` to `action_to`, excluding the start point and including (or stopping near) the end point. Must be non-blocking and pure.

**Why.** Each robot has its own action layout — what is a joint angle vs. a normalized finger value, where wrap-around matters, what the per-dimension clip ranges are. Pushing the interpolation policy into the mux would make it robot-aware. The abstraction lets `ActionMux` stay robot-agnostic and shoves the per-robot knowledge into a small file that lives next to the robot's other config.

### `interp_utils/robots/__init__.py` — renamed (from `io_interface/__init__.py`)

**Change.** Git tracks this as a rename of the now-deleted `env_actor/human_in_the_loop/io_interface/__init__.py` (R100, content unchanged: empty package marker).

**Why.** It is an empty `__init__.py`; git's rename heuristic chose an unrelated deleted file as its "source" because empty files all match each other. Treat it as a brand-new package marker for the interpolator subpackage.

### `interp_utils/robots/igris_b/__init__.py` — new

**Change.** Re-exports `IgrisBInterpolator` so callers can write `from .interp_utils.robots.igris_b import IgrisBInterpolator`.

**Why.** Convenience import path; without it, the mux's `_build_interpolator` would have to know the file name.

### `interp_utils/robots/igris_b/igris_b_interpolator.py` — new

**Change.** Concrete igris_b implementation ([`igris_b_interpolator.py:20-49`](../../env_actor/human_in_the_loop/action_mux/interp_utils/robots/igris_b/igris_b_interpolator.py#L20-L49)). The 24-D action is `[0:12]` arm joint angles in radians + `[12:24]` finger targets normalized to `[0, 1]`. Joints use shortest-arc interpolation; fingers use clipped linear interpolation.

**Why.** Naive linear interpolation between, for instance, `+170°` and `-170°` would take the long way around through `0°`, mechanically wrong and a likely cable-tangle. Shortest-arc respects the angular topology. Fingers are not angle-valued so they can be linearly interpolated, but they must be clipped to `[0, 1]` because a small numerical drift outside that range causes the actuator to interpret the value as "open all the way" or "close all the way." See [10 Q16](./10_faq_onboarding.md#q16-the-action-mux-uses-shortest-arc-interpolation-for-joints-why) for the wrap-around walk-through.

### `intervention_switch.py` — modified

**Change.** Pedal event semantics simplified ([`intervention_switch.py:53-62`](../../env_actor/human_in_the_loop/action_mux/intervention_switch.py#L53-L62)):

- `$` → set mode to TELEOP, log transition.
- `#` → set mode to POLICY, log transition.

Previously `$` toggled and `^` was the policy edge.

**Why.** Toggling is not idempotent: if the pedal driver dropped or duplicated an event (BEST_EFFORT QoS used to allow this — see the pedal_subscriber change), the toggle would leave the mode in the wrong state and the only fix was to press the pedal again until it agreed. The new edges are absolute commands — pressing the TELEOP pedal twice keeps the system in TELEOP, never accidentally toggles it back to POLICY. The print is for operator feedback when there is no visual indicator.

### `teleop_provider.py` — modified

**Change.** `IgrisBTeleopProvider.__init__` no longer requires an external `SingleThreadedExecutor`. It creates two executors (one for `DxlMasterArm`, one for `ManusUDPReceiver`) and starts each on its own daemon thread ([`teleop_provider.py:46-78`](../../env_actor/human_in_the_loop/action_mux/teleop_provider.py#L46-L78)). `shutdown()` shuts both executors before destroying the nodes ([`teleop_provider.py:82-87`](../../env_actor/human_in_the_loop/action_mux/teleop_provider.py#L82-L87)).

**Why.** On `main`, the control loop's executor spun the master arm, the glove, *and* the rest of the ROS pubsub. If the glove receiver missed a UDP packet or stalled, the master arm read could not happen for the same tick, causing the control loop to receive `None` and fall back to policy. Per-node executors decouple the hardware: each runs at its own cadence and a slow node does not starve the others. The two daemon threads die with the process so there is no shutdown bookkeeping for the caller.

## `env_actor/human_in_the_loop/inference_algorithms/rtc/`

This subtree was substantially restructured. The previous layout:

```
rtc/
  control_actor.py            (Ray remote)
  inference_actor.py          (Ray remote)
  inference_engine_utils/
    action_inpainting.py
    max_deque.py
  data_manager/
    data_normalization_interface.py
    robots/igris_b/{data_normalization_manager.py, shm_manager_bridge.py}
    shm_manager_interface.py
    utils/utils.py
```

The new layout:

```
rtc/
  rtc_actor.py                (Ray remote; spawns child processes)
  actors/
    control_loop.py
    inference_loop.py
  data_manager/
    robots/igris_b/shm_manager_bridge.py
    shm_manager_interface.py
    utils/{max_deque.py, shared_memory_utils.py}
```

**Why the restructure.** On `main`, the HIL RTC tree had *two* Ray remotes (one per actor) while the auto RTC tree had *one* Ray remote that spawned two `multiprocessing` children. Two completely different supervision models for the same algorithm cost engineering time every time someone fixed a bug — fixes had to be ported twice and silent divergence was the norm. Bringing HIL in line with the auto layout (one Ray actor + two child processes + shared memory) means: one supervision model, one set of synchronization primitives, one episode-boundary refactor that applies to both. The `data_normalization_*` files are deleted because normalization moved into the policy itself (see `sequential_actor.py` rationale below).

### `rtc_actor.py` — new

**Change.** Single Ray actor that spawns two `multiprocessing` child processes with `spawn` start method, attaches them to shared memory, and supervises lifetime ([`rtc_actor.py:1-188`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/rtc_actor.py#L1-L188)). Same shape as the auto-side actor, plus residual flags.

**Why.** See the "Why the restructure" preamble. The Ray actor is the supervision boundary that survives child crashes; the `multiprocessing` children are where the real work happens, because shared memory (proprio / image / action) cannot cross Ray actor boundaries cheaply.

### `actors/control_loop.py` — new

**Change.** Mirrors the auto control loop, plus:

- Constructs `IgrisBTeleopProvider`, `PedalInterventionSwitch`, and `ActionMux` ([`hil/.../control_loop.py:93-104`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py#L93-L104)).
- After each control step: `policy_action = shm.atomic_write_obs_and_increment_get_action(...)` → `action, control_mode = action_mux.select(policy_action)` → if `use_residual_rl` and `control_mode != TELEOP`, apply the residual; if TELEOP, pass through unchanged ([`hil/.../control_loop.py:213-231`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py#L213-L231)).
- At episode end, `action_mux.set_control_mode_to_policy()` resets the mux to POLICY and discards any in-flight interpolation ([`hil/.../control_loop.py:150-151`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py#L150-L151)).
- Cleanup calls `teleop_provider.shutdown()` before `rclpy.shutdown()`.

**Why each sub-change.**

- **Mux owned by the control loop, not the actor.** The mux is per-tick state; spawning it inside the Ray actor would force it to cross the actor → process boundary on every call, which is expensive and brittle. Living in the control-loop process gives it direct access to the same ROS node and the same numpy arrays as the action publisher.
- **Skip residual when TELEOP.** When the operator is in control, the policy's action is irrelevant; applying a residual on top of a teleop pose would corrupt the operator's command. The conditional preserves the operator's intent and keeps the buffer clean (the `control_mode=TELEOP` rows still flow to the buffer; the HIL-split buffer can route them away from the off-policy critic if you enable it).
- **Reset mux at episode end.** A leftover POLICY→TELEOP trajectory from a previous episode would otherwise emit stale steps at the start of the next episode. Resetting the mux to POLICY plus dropping the in-flight trajectory ensures every new episode starts from a clean state.
- **`teleop_provider.shutdown()` before `rclpy.shutdown()`.** The teleop nodes own ROS resources. Tearing down ROS first leaves the nodes in an invalid state and triggers warnings (or worse, hangs) during garbage collection.

### `actors/inference_loop.py` — new

**Change.** Same as the auto-side inference loop ([`hil/.../inference_loop.py`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/inference_loop.py)).

**Why.** There is nothing HIL-specific about the inference loop — it runs the base policy, writes chunks to shared memory, and gates weight reload on `use_residual_rl`. Keeping it as a near-copy of the auto version makes "test fix on auto then port to HIL" trivial. Duplicate code is the lesser evil here vs. a shared module that would couple two trees and break the symmetry.

### `__init__.py` — modified

**Change.** Re-exports `RTCActor` (single name) instead of `InferenceActor` + `ControllerActor`; imports come from `data_manager/utils/shared_memory_utils.py` instead of `data_manager/utils/utils.py` ([`__init__.py:13-23`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/__init__.py#L13-L23)).

**Why.** The package's public surface should reflect the new structure. Leaving the old exports would tempt callers to import the deleted Ray remotes.

### Deletions inside `rtc/`

| Deleted file | Why deleted |
|---|---|
| `control_actor.py` (245 lines) | Replaced by `actors/control_loop.py` (a function the rtc_actor process pool calls), not a Ray remote. The Ray-remote-per-actor pattern was abandoned (see preamble above). |
| `inference_actor.py` (194 lines) | Same reason — replaced by `actors/inference_loop.py`. |
| `inference_engine_utils/action_inpainting.py` | The canonical copy lives at [`env_actor/inference_engine_utils/action_inpainting.py`](../../env_actor/inference_engine_utils/action_inpainting.py). The HIL duplicate was already drifting in small ways from the auto version and was deleted to avoid bug-fix-twice maintenance. |
| `inference_engine_utils/__init__.py` | The package became empty after `action_inpainting.py` left; an empty package wastes import lookups. |
| `data_manager/data_normalization_interface.py` | Normalization moved inside the policy (see `sequential_actor.py` rationale below). The data manager no longer needs to know about norm stats. |
| `data_manager/robots/igris_b/data_normalization_manager.py` | Same reason — the per-robot normalization bridge is obsolete once normalization is inside the policy. |

### Renames

| Old path | New path | Why renamed |
|---|---|---|
| `data_manager/utils/utils.py` | `data_manager/utils/shared_memory_utils.py` | The old name was uninformative — three "utils" tokens in a row. The new name says what the module is for. |
| `inference_engine_utils/max_deque.py` | `data_manager/utils/max_deque.py` | `MaxDeque` is consumed by the shared-memory bridge to track per-iteration counters, so it belongs next to the bridge, not under the now-deleted `inference_engine_utils` subtree. |

### `data_manager/robots/igris_b/shm_manager_bridge.py` — modified

**Change.** Same role as on `main`, with four small but important changes:

- Imports `MaxDeque` from the new location ([`shm_manager_bridge.py:27`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py#L27)).
- `set_inference_ready` / `set_inference_not_ready` now hold the `inference_ready_cond` lock only — the previous double-lock pattern is gone ([`shm_manager_bridge.py:180-198`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py#L180-L198)).
- `atomic_write_obs_and_increment_get_action` clamps the action index to `≥ 0` ([`shm_manager_bridge.py:294`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py#L294)).
- Adds `init_action_chunk_obs_history` ([`shm_manager_bridge.py:367-394`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py#L367-L394)).

**Why each.**

- **Import path** — `MaxDeque` moved (see Renames).
- **Lock simplification** — taking `self._lock` and `self._inference_ready_cond` in two separate `with` blocks created a window where a waiter could miss the `notify_all`. Holding the condition's lock only is enough because the flag is a simple `bool` and only ever read inside the condition's `with`.
- **Action-index clamp** — on the very first tick `num_control_iters == 0`, so `min(iters - 1, chunk_size - 1) == -1` indexed the last element of the chunk (numpy negative indexing), silently corrupting the action. Clamping with `max(0, ...)` makes the off-by-one harmless.
- **`init_action_chunk_obs_history`** — same rationale as the auto-side counterpart: bundle the episode-boundary reset into one atomic call.

### `data_manager/shm_manager_interface.py` — modified

**Change.** Adds `init_action_chunk_obs_history` proxy ([`shm_manager_interface.py:124-127`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/data_manager/shm_manager_interface.py#L124-L127)).

**Why.** Same as the auto-side proxy — the interface keeps a flat public API so callers do not reach through it.

## `env_actor/human_in_the_loop/inference_algorithms/sequential/`

### `sequential_actor.py` — modified

**Change.** Substantial cleanup. Imports now resolve to `env_actor.robot_io_interface.controller_interface.ControllerInterface` and `env_actor.auto.inference_algorithms.sequential.data_manager.robots.igris_b.data_manager_bridge.DataManagerBridge` ([`sequential_actor.py:60-66`](../../env_actor/human_in_the_loop/inference_algorithms/sequential/sequential_actor.py#L60-L66)). The teleop wiring (master arm + glove + pedal mux) is removed from this actor. The constructor signature changed from `runtime_params` to `inference_runtime_params_config` (raw config dict/path).

**Why.**

- **Canonical imports.** Before this branch, the HIL package shipped its own copy of `controller_interface` (in `io_interface/`) that drifted from the canonical one in `robot_io_interface/`. Re-pointing imports to the canonical version is what makes deleting `io_interface/` safe. Same for the data_manager — the HIL copy has been superseded by the auto copy.
- **Teleop removed from sequential.** Teleop now lives only in the RTC HIL control loop, where the `ActionMux` belongs. Sequential mode is reserved for non-teleop debugging (e.g. running a single-shot policy without the RTC chunking complexity); keeping teleop in sequential as well would force the actor to import master-arm + glove + pedal modules even when they are unused, which wastes startup time and complicates failure modes.
- **Constructor signature change.** Passing a raw config path/dict instead of a constructed `RuntimeParams` means the caller does not need to know about `RuntimeParams`'s constructor signature, and the actor can lazily import the robot-specific class. This matches how `RTCActor` is invoked on this branch.

### `data_manager/data_manager_interface.py` — modified

**Change.** Stripped to the methods residual flow needs: `update_state_history`, `buffer_action_chunk`, `get_current_action`, `init_inference_obs_state_buffer`, `serve_raw_obs_state`, `serve_init_action`. Removed: `prev_joint` property, `denormalize_action`, `normalize_action_chunk`, `update_prev_joint`, `update_norm_stats`, `serve_normalized_obs_state`, `generate_noise`, `get_raw_obs_arrays`.

**Why.** Normalization, noise generation, and joint-state tracking moved into other components. Specifically:

- Normalization is now inside the policy (see [`resfit_policy.py`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py) and the DSRL-OpenPI policy on the auto path). Keeping a parallel set of normalization methods on the data manager invited mismatches where the same observation could be normalized twice or with different stats.
- `prev_joint` tracking moved to the sequential actor itself ([`sequential_actor.py:181`](../../env_actor/human_in_the_loop/inference_algorithms/sequential/sequential_actor.py#L181)). The actor was always the only caller, so the indirection was redundant.

The smaller interface reduces the surface a maintainer has to reason about when adding a new robot.

### `data_manager/robots/igris_b/data_manager_bridge.py` — modified

**Change.** Re-pointed `from env_actor.runtime_settings_configs.igris_b...` → `from env_actor.runtime_settings_configs.robots.igris_b...`. Stripped per-bridge normalization. The methods that survive are buffer-management primitives only.

**Why.** Symmetric with the interface change: the bridge does not need to know about norm stats anymore. The import re-point catches up with the directory restructure that landed on `main` earlier.

### `data_manager/robots/igris_c/data_manager_bridge.py` — modified

**Change.** One-line comment fix: `/env_actor/auto/data_manager/igris_b/...` → `.../robots/igris_b/...`.

**Why.** The reference path in the doc comment was stale after the auto-package restructure. Fixing it costs nothing and prevents the next person from following the dead path.

## `env_actor/human_in_the_loop/intervention_methods/pedal/`

### `publisher/pedal_publisher.py` — modified

**Change.** Default `robot_id` changed from `'cashy'` → `'packy'` ([`pedal_publisher.py:18-19`](../../env_actor/human_in_the_loop/intervention_methods/pedal/publisher/pedal_publisher.py#L18-L19)).

**Why.** `cashy` and `packy` are the two physical pedal hubs in the lab. The pedal driver had been moved from one to the other (commit `f0c2357 updated pedal publisher to connect to igris_b packy instead of cashy`). Updating the default makes the script "just work" on the rig where it is most often run. If you are bringing up a new rig, override on the CLI.

### `subscriber/pedal_subscriber.py` — modified

**Change.** The ready publisher's QoS changed from `BEST_EFFORT` → `RELIABLE` ([`pedal_subscriber.py:1139-1144`](../../env_actor/human_in_the_loop/intervention_methods/pedal/subscriber/pedal_subscriber.py#L1139-L1144)).

**Why.** Episodes start when both ends agree "ready." Under load, `BEST_EFFORT` silently dropped the "ready" message, leaving the system in a state where the rollout would not start and there was no error message. `RELIABLE` forces ROS 2 to retransmit until acked; the cost is a tiny amount of bandwidth, well worth the disappeared "why won't training start?" class of bug. Pairs with the idempotent pedal-edge change ([`intervention_switch.py`](#intervention_switchpy--modified)).

## `env_actor/human_in_the_loop/io_interface/` (deleted)

**Change.** The branch deletes:

- `controller_interface.py`
- `robots/igris_b/{__init__.py, controller_bridge.py, utils/{__init__.py, camera_utils.py, data_dict.py}}`
- `robots/igris_c/{__init__.py, controller_bridge.py, igris_c.py, utils/__init__.py}`

The HIL packages now import the canonical versions from `env_actor/robot_io_interface/`. See the new imports in [`sequential_actor.py:60`](../../env_actor/human_in_the_loop/inference_algorithms/sequential/sequential_actor.py#L60) and [`hil/.../control_loop.py:39`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py#L39).

**Why.** Every file in this subtree was a near-duplicate of a file in `env_actor/robot_io_interface/`. Maintaining two copies meant every bug fix had to be applied twice and was easily forgotten on the HIL side, where rollout volume is lower so bugs surfaced later. Deleting the duplicates forces the HIL path to use the same controller code as the auto path; behavior is identical by construction, not by review.

If your local branch imports from `env_actor/human_in_the_loop/io_interface/`, re-point to `env_actor/robot_io_interface/` — same class names.

## `env_actor/human_in_the_loop/teleoperation/`

### `robots/igris_b/arms_dynamixel.py` — modified

**Change.** Two robustness changes:

- **Ping at startup.** After opening the port, every motor ID is pinged up to three times; if any fails, the constructor raises with the offending ID ([`arms_dynamixel.py:131-144`](../../env_actor/human_in_the_loop/teleoperation/robots/igris_b/arms_dynamixel.py#L131-L144)).
- **Tolerant sync-read.** `sync_read_position` keeps `_sync_read_fail_count`. A single failure is logged as a warning and the previous positions are kept; the error is only raised after 10 consecutive failures, with the list of unresponsive IDs ([`arms_dynamixel.py:170-189`](../../env_actor/human_in_the_loop/teleoperation/robots/igris_b/arms_dynamixel.py#L170-L189)).

**Why.**

- **Ping.** Previously, a loose cable would manifest as a `sync_read_position` failure mid-training, after hours of rollouts had already accumulated. Pinging at startup catches the cable issue before anyone has to wait. Failing fast with the exact motor ID also tells the operator which cable to wiggle.
- **Tolerant sync-read.** A single dropped USB packet would raise and kill the entire control-loop process, taking down the training run. In practice, motors are accurate enough that "keep previous positions for one frame" is a perfectly fine fallback. Ten consecutive failures *is* a real hardware problem (the kind ping would have caught at startup if it had been transient at boot), so the eventual raise is still there as a circuit breaker.

## `env_actor/policy/policies/`

### `dsrl_openpi_policy/components/noise_actor.yaml` — modified

**Change.** `input_dim: 49584` → `input_dim: 6960` ([`noise_actor.yaml:6-7`](../../env_actor/policy/policies/dsrl_openpi_policy/components/noise_actor.yaml#L6-L7)).

**Why.** The DSRL-OpenPI architecture went through two image-backbone variants during base-policy development. The 49 584-D variant assumed an OpenPI-default 1 024-channel feature extractor; the 6 960-D variant assumes the ResNet34Group used at training time (512-channel block-4 output, 1×1-conv-projected to 24 channels per camera, three cameras, plus proprio + noise — `3 × (24 × 8 × 10) + 24 + 1128 = 6960`). Without matching, `load_state_dict` raises a shape mismatch on the very first forward pass.

### `dsrl_openpi_policy/components/noise_processor.yaml` — modified

**Change.** `input_img_channel: 1024` → `input_img_channel: 512` ([`noise_processor.yaml:11`](../../env_actor/policy/policies/dsrl_openpi_policy/components/noise_processor.yaml#L11)).

**Why.** Same as above — the preprocessor must agree with the backbone it sits on top of. ResNet34's penultimate stage emits 512 channels, so the 1×1 projection's `in_channels` must be 512.

### `dsrl_openpi_policy/components/openpi_model.yaml` — modified

**Change.** `default_prompt` changed from "Use the left hand…" → "Use the right hand…".

**Why.** The DSRL-OpenPI checkpoint used during residual-RL training was fine-tuned with right-hand-prompted episodes. Sending it the left-hand prompt would push the action distribution into the part of state space it has seen *least*, which is exactly the wrong starting condition for residual fine-tuning.

### `resfit_policy/__init__.py` — renamed (from `io_interface/robots/igris_b/__init__.py`)

**Change.** Tracked as a rename by git (R100, empty file). Functionally a new package marker for the resfit policy subpackage.

**Why.** Same as other empty-`__init__.py` "renames" in this branch — git's rename detection picks whichever deleted empty file matches. Treat as new.

### `resfit_policy/resfit_policy.yaml` — new

**Change.** Top-level policy YAML registering `resfit_policy` with the component path and constructor params ([`resfit_policy.yaml`](../../env_actor/policy/policies/resfit_policy/resfit_policy.yaml)).

**Why.** The env_actor's policy loader is YAML-driven (see [`docs/05_policy_protocol.md`](../05_policy_protocol.md)). To register a new policy, you need one of these files. Without it, `build_policy` cannot find the residual class.

### `resfit_policy/components/resfit_residual_actor.yaml` — new

**Change.** Builds `Residual_Actor` — Resnet34Group + ResidualActorPreprocessor + 5-layer 2048-dim MLP + Tanh × 0.25 ([`resfit_residual_actor.yaml`](../../env_actor/policy/policies/resfit_policy/components/resfit_residual_actor.yaml)).

**Why.** GraphModel architecture is described in YAML so non-Python clients (and future model swaps) do not require code changes. The fields here must exactly match the trainer-side YAML at [`experiment_models/resfit/exp1/resfit_residual_actor.yaml`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_models/resfit/exp1/resfit_residual_actor.yaml), otherwise `load_state_dict` will fail when the inference side tries to load training-side weights.

### `resfit_policy/resfit_policy.py` — new

**Change.** `ResidualPolicy` class ([`resfit_policy.py:29`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py#L29)), registered under `resfit_policy` in `POLICY_REGISTRY`. Implements:

- `inference(base_action, obs_data)` — runs the residual actor once, returns the residual delta. Used by the control loop.
- `predict(input_data, data_normalization_interface)` — Policy-protocol entry point. Returns `base + residual`.
- `guided_inference(...)` — protocol-conformant blend, currently unused.
- `warmup`, `eval`, `to`, `parameters`, `state_dict`, `load_state_dict`, `freeze_all_model_params` — Policy-protocol boilerplate.

The constructor optionally loads `<checkpoint_path>/resfit_residual_actor.pt` ([`resfit_policy.py:52-56`](../../env_actor/policy/policies/resfit_policy/resfit_policy.py#L52-L56)).

**Why.** The env_actor loader expects every policy to satisfy a small protocol (see [`docs/05_policy_protocol.md`](../05_policy_protocol.md)). Implementing the protocol means the residual policy can be loaded and supervised by the same machinery that loads the base policy — no special-case code in the loader. The lightweight `.inference(base_action, obs_data)` is what the control loop actually calls hot-path; it bypasses `predict` because the control loop already has the base action and does not need the protocol-mandated dispatch on a normalization interface.

## `env_actor/robot_io_interface/`

### `controller_interface.py` — modified

**Change.** Adds a `ros_node` accessor ([`controller_interface.py:14-17`](../../env_actor/robot_io_interface/controller_interface.py#L14-L17)) proxying to `self.controller_bridge.ros_node`.

**Why.** The HIL RTC control loop hands the controller's existing ROS node to `PedalInterventionSwitch` so the pedal subscriber piggybacks on it. Without an accessor, callers would have to reach `self.controller_bridge.input_recorder` directly — that breaks the bridge's encapsulation and gives the pedal subscriber an implementation-detail dependency it does not need.

### `robots/igris_b/controller_bridge.py` — modified

**Change.** Same `ros_node` accessor on the bridge ([`controller_bridge.py:51-53`](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py#L51-L53)). The bridge's `shutdown()` no longer calls `rclpy.shutdown()`.

**Why.**

- **Accessor.** Symmetric to the interface — the interface proxies, the bridge implements.
- **`rclpy.shutdown()` removal.** The HIL RTC control loop now calls `rclpy.shutdown()` explicitly after tearing down the teleop provider ([`hil/.../control_loop.py:257`](../../env_actor/human_in_the_loop/inference_algorithms/rtc/actors/control_loop.py#L257), [`auto/.../control_loop.py:230`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L230)). If the bridge also called it during `shutdown()`, the second call would log "rclpy already shutdown" warnings and, on some ROS 2 distros, deadlock the executor during teardown. Calling it exactly once, at the end of the loop, makes the lifecycle explicit.

## Root-level files

### `run_online_rl.py` — modified

**Change.** The single most important entry-point change. Summary:

- New CLI flags: `--use_residual_rl`, `--residual_policy_yaml`, `--use_human_intervention`, default `--num_labeler_gpus=4`.
- Imports `ResfitReplayBufferActor` instead of `ReplayBufferActor`.
- Instantiates the buffer with `action_horizon=4, reward_horizon=3, obs_subsample_step=3` ([`run_online_rl.py:88-93`](../../run_online_rl.py#L88-L93)).
- Picks the HIL `RTCActor` when `--use_human_intervention` is set, else the auto one. Same for sequential.
- Forwards `residual_policy_yaml_path` and `use_residual_rl` into the chosen actor.
- The `@ray.remote` decorator on `run_training` is commented out; `run_training` is invoked synchronously in the driver process ([`run_online_rl.py:176`](../../run_online_rl.py#L176)).
- `RAYQUEUE_MAXSIZE` reduced from `15` → `5`.
- `TorchTrainer` `num_workers` reduced from `4` → `1`; binds `resources_per_worker={"training_pc": 1}`.

**Why each.**

- **New CLI flags.** Each flag toggles one orthogonal feature (residual on/off, residual YAML choice, teleop on/off). Making them flags rather than YAML fields lets you smoke-test the *exact* same configs across modes by changing only the CLI.
- **`ResfitReplayBufferActor`.** Residual RL needs the new buffer schema (see `resfit_replay_buffer.py` rationale). Importing the new class is the smallest change at the entry point.
- **`action_horizon=4, reward_horizon=3, obs_subsample_step=3`.** These are not the *class defaults* — the class default is `action_horizon=50` to keep the API self-contained. The script overrides match the trainer YAML at [`resfit.yaml: data.datamodule.params`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml#L41-L55). Mismatching causes a silent shape error when offline and online batches are concatenated.
- **HIL vs auto branch.** Same algorithm, different teleop wiring. Choosing the import path at the driver lets the rest of the code stay agnostic.
- **Synchronous `run_training`.** Wrapping `run_training` as a `@ray.remote` task on top of `TorchTrainer` (which itself uses `ScalingConfig` to spawn Ray workers) was a layer of indirection that did not buy anything — the outer Ray task did nothing but call `.fit()`. Removing it simplifies the driver lifecycle and removes one layer of Ray scheduling. The commented-out lines just above ([`run_online_rl.py:141-146`](../../run_online_rl.py#L141-L146)) document the previous form for archaeology purposes; see [10 Q11](./10_faq_onboarding.md#q11-why-is-run_training-not-a-ray-task-anymore).
- **`RAYQUEUE_MAXSIZE` 15→5.** A queue this size buffers about 5 episodes ≈ 2 minutes of rollouts. Letting 15 episodes accumulate consumed too much memory on the head node when the labeler was slow; smaller queue gives explicit back-pressure into the control loop's drop-with-warning rather than silent memory bloat.
- **`num_workers=1`.** The current training PC has one GPU available for the trainer (the others are reserved by labelers and inference). Spawning four DDP workers on one GPU competes for memory and deadlocks; one worker matches hardware. `resources_per_worker={"training_pc": 1}` is required for Ray's resource scheduling to place that one worker on the right machine.

### `start_ray.sh` — modified

**Change.** Per-node symbolic resource quotas raised from `{labeling_pc:4, training_pc:3, inference_pc:1}` → `100` each ([`start_ray.sh:18-39`](../../start_ray.sh#L18-L39)).

**Why.** Ray resource accounting refuses to schedule an actor that would exceed a node's declared quota. The old quotas were sized for one specific config (four auto labelers, three training workers, one inference actor). Any reconfiguration — more labelers, two-process inference, multi-worker training — failed with `No available resources` errors that were misleading. Setting the quotas to 100 each turns the labels into pure placement constraints (which machine runs the actor) rather than capacity caps; the actual hardware is the real ceiling, as it should be.
