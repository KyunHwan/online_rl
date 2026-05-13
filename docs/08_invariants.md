# 08 — Invariants

These are the architectural rules that any change to the pipeline must respect. Breaking one will not always crash immediately — sometimes the system runs but produces wrong actions or silently desyncs. Each rule names what it is, why it exists, and what fails when it is violated.

## Normalization is inside the policy

**Rule.** The policy accepts a `DataNormalizationInterface` and calls it internally. No component outside the policy normalizes observations or denormalizes actions.

**Why.** Two reasons. (1) The trainer learns on normalized data, so the policy network expects normalized inputs and produces normalized outputs. Putting normalization inside the policy means the network's API matches what was trained. (2) Different policy components may have different normalization needs (e.g., the DSRL policy's `backbone` receives normalized proprio while the OpenPI submodel receives raw proprio — it manages its own stats). The policy is the only place that knows this.

**What breaks.** If you normalize in the control loop, every new policy you swap in has to assume the same normalization scheme. If you forget to denormalize, the robot moves toward zero-mean joint targets that look nothing like trained behaviors.

**Where to verify in code.** [env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py) — see `_run_inference()` calling `data_normalization_interface.normalize_state(...)` early.

## All data crossing the policy boundary is numpy

**Rule.** Input dicts to `predict()` and `guided_inference()` contain only `np.ndarray` values. Output is a `np.ndarray`. Inside the policy, numpy → torch and torch → numpy conversions happen, but no torch tensor escapes the policy's call boundary.

**Why.** The transport machinery on both sides of the policy is numpy-native. Shared memory holds raw numpy buffers; `EpisodeRecorderBridge.add_obs_state()` calls `torch.from_numpy()` on numpy inputs; `ControllerBridge.publish_action()` expects numpy and would crash on a CUDA tensor. Forcing the policy boundary to numpy means none of those callers have to know about device placement.

**What breaks.** Return a `torch.cuda.Tensor` from `predict()` and the shared-memory writer (RTC) or the control loop's downstream code crashes with `astype` or `concatenate` errors. Pass a `torch.Tensor` into `predict()` and `torch.from_numpy()` later fails with `TypeError`.

**Where to verify in code.** Search for `cpu().float().numpy()` in [env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py) — every torch tensor is converted before return.

## The `DataNormalizationInterface` uses only numpy

**Rule.** The normalization manager has zero torch imports. Stats are numpy arrays; inputs and outputs are numpy arrays.

**Why.** The normalizer runs in the same process as the policy, but the boundary is numpy (see above). If the normalizer needed torch, every consumer of `normalize_state()` would have to think about device placement. Keeping it numpy lets the normalizer live anywhere — in the env actor process, in a unit test, on a CPU-only machine — without setup ceremony.

**What breaks.** If you import torch into a normalizer bridge, the unit test that imports `DataNormalizationInterface` on a CPU-only laptop suddenly needs CUDA-aware torch.

**Where to verify in code.** [env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py](../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py) — `import numpy as np`, no `import torch`.

## The policy exposes its modules via `self.components`

**Rule.** Every `nn.Module` that the trainer should be able to update online must be in `self.components`, under the same key the trainer uses in `trainer.models`. The policy stores the components dict from its constructor as the attribute literally named `components`.

**Why.** The inference loop's weight update is:

```python
for model_name in current_weights.keys():
    if model_name in policy.components.keys():
        load_state_dict_cpu_into_module(policy.components[model_name], current_weights[model_name], strict=True)
```

The trainer's push side iterates `trainer.models.keys()` and only pushes components flagged `online_update: true` in the trainer YAML. The key intersection is the contract.

**What breaks.** Rename `self.components` → `self.modules_` and the inference loop silently never updates weights. Don't put a model in `self.components` and the trainer's weight push is ignored — the inference loop keeps using initial weights for that module.

**Where to verify in code.** [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) lines 102–108 and [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 536.

## Guided inference (action inpainting) lives inside the policy

**Rule.** `guided_inference()` does its own blending between the previous action chunk's tail and the new prediction. The RTC inference loop does **not** post-process the policy output; it writes whatever the policy returned straight back to shared memory.

**Why.** Different policies have different blend semantics — a flow-matching policy might do the inpainting during the ODE solve ([guided_action_chunk_inference()](../env_actor/inference_engine_utils/action_inpainting.py) is exactly that), while a simpler policy does post-hoc blending. Keeping the blend inside the policy lets each policy class pick its own approach without touching the surrounding loop.

**What breaks.** Move the blend into the inference loop and you have hardcoded one specific schedule, dtype, and shape assumption for every future policy. Skip the blend entirely (return only the new prediction) and you see action discontinuities every time inference runs — the robot will jerk.

**Where to verify in code.** Both production policies' `guided_inference()` methods do the blend themselves; [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) calls `policy.guided_inference(...)` and writes the result unchanged.

## Episode flow uses `ray.put` + Ray Queue, not direct passing

**Rule.** `EpisodeRecorderBridge.serve_train_data_buffer()` returns a list of TensorDicts; the env-actor side puts each into the Ray [Plasma object store](10_glossary.md#plasma-object-store) with `ray.put(td)` and enqueues the `ObjectRef`. The labeler dequeues the ref, `ray.get`s the TensorDict, and pushes the labeled TensorDict into the replay buffer.

**Why.** A TensorDict for a 1000-step episode is large (hundreds of MB once cameras are stacked). Passing it through a Ray actor's RPC argument would serialize that whole blob on every call. Plasma keeps it in shared memory; the labeler reads from the same memory page the env actor wrote to.

**What breaks.** Calling `replay_buffer_actor.add.remote(td)` directly from the env actor side-steps the labeler and the rate limit, plus serializes the whole TensorDict over RPC. The labeler is exactly where reward is generated — bypassing it means episodes get into the replay buffer with `reward = 0` everywhere.

**Where to verify in code.** [env_actor/auto/inference_algorithms/rtc/actors/control_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py) — `sub_ep_data_ref = ray.put(sub_ep); episode_queue_handle.put(sub_ep_data_ref, block=True)`.

## Weights flow via the StateManager named actor

**Rule.** The trainer pushes weight refs into `StateManagerActor` (looked up by the name `"policy_state_manager"`). The inference loop polls `StateManagerActor.get_state()` between episodes. Version counters gate updates so each push is consumed at most once.

**Why.** Three reasons. (1) Decouples training cadence from inference cadence — the trainer pushes every N iterations, the inference loop checks once per episode boundary. (2) Plasma-backed transfer avoids serializing weights through RPC: `ray.put(state_dict)` puts the dict in Plasma; only the ref crosses the wire. (3) The named actor pattern means any process on the cluster can find it; you do not have to thread handles through every function call.

**What breaks.** Rename the actor in [run_online_rl.py](../run_online_rl.py) without updating `ray.get_actor(...)` calls in the trainer and the RTC inference loop, and the trainer's next checkpoint push will fail with `ValueError: Failed to look up actor`. The Sequential actor will keep running because it receives the handle directly — that asymmetry has caused real bugs.

**Where to verify in code.** [data_bridge/state_manager.py](../data_bridge/state_manager.py) (the actor), [run_online_rl.py](../run_online_rl.py) line 80 (where the name is set), [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) line 94 (RTC lookup), [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 424 (trainer lookup).

## RTC's two processes communicate through SharedMemory + multiprocessing primitives, not through Ray

**Rule.** Inside the RTC actor, the inference process and the control process communicate exclusively via `multiprocessing.shared_memory.SharedMemory` (for data) and `multiprocessing.RLock` / `Condition` / `Event` / `Value` (for synchronization). The two processes are forked by `mp.get_context("spawn")`, not by Ray. The Ray Queue and the `StateManagerActor` are the only places the two processes touch Ray.

**Why.** Two reasons. (1) Latency — Ray RPC is fast but not 20 Hz fast when you add up the round trips of a polling control loop. SharedMemory is zero-copy. (2) GIL — even if Ray's overhead were acceptable, running both loops in one Python process means the GPU forward pass blocks the control loop's `time.sleep` accounting.

**What breaks.** Try to share state via a Ray actor between the two loops and you eat one or two Ray RPCs per 50 ms control step, which budgets blow.

**Where to verify in code.** [env_actor/auto/inference_algorithms/rtc/rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) — see the `ctx = mp.get_context("spawn")` block. Locks and Conditions are passed to both children.

---

Next: [09_troubleshooting.md](09_troubleshooting.md) — what to check when one of these invariants gets violated.
