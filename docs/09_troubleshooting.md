# 09 — Troubleshooting

Symptom-first. Find your error message or behaviour, jump to the section, follow the fix.

## Table of contents

- [Cluster bring-up](#cluster-bring-up)
  - [ray start fails / Tailscale not found / nodes can't see each other](#ray-start-fails--tailscale-not-found--nodes-cant-see-each-other)
  - [Too many open files / ulimit errors](#too-many-open-files--ulimit-errors)
- [Install and import](#install-and-import)
  - [ImportError or weights not loading: transformers patch](#importerror-or-weights-not-loading-transformers-patch)
  - [Submodule missing](#submodule-missing)
- [First-run config errors](#first-run-config-errors)
  - [norm_stats_file_path: File not found](#norm_stats_file_path-file-not-found)
  - [--robot igris_c crashes immediately](#--robot-igris_c-crashes-immediately)
  - [--train_config path doesn't exist](#--train_config-path-doesnt-exist)
- [Runtime behaviour](#runtime-behaviour)
  - [Replay buffer never produces samples / sample shape is wrong](#replay-buffer-never-produces-samples--sample-shape-is-wrong)
  - [Inference uses stale weights forever](#inference-uses-stale-weights-forever)
  - [GUI labeler shows blank frames](#gui-labeler-shows-blank-frames)
  - [Why isn't my pedal/teleop doing anything when I pass --human_reward_labeler?](#why-isnt-my-pedalteleop-doing-anything-when-i-pass---human_reward_labeler)
  - [Manual reward labeler never pumps the queue](#manual-reward-labeler-never-pumps-the-queue)
  - [Auto labeler cannot find task](#auto-labeler-cannot-find-task)
  - [Episode queue full, env actor stalls](#episode-queue-full-env-actor-stalls)
  - [Policy weights silently never update](#policy-weights-silently-never-update)

---

## Cluster bring-up

### ray start fails / Tailscale not found / nodes can't see each other

**Symptom.** `start_ray.sh` aborts with `Error: Could not find a Tailscale IP. Is Tailscale running?`, or `ray status` from one machine doesn't show the others.

**Where to look.** [start_ray.sh](../start_ray.sh) lines 1–15. The script needs `tailscale ip -4` to return something; if Tailscale isn't installed or running, it bails. The `HEAD_IP="100.118.28.46"` constant is deployment-specific.

**Fix.**

1. `tailscale ip -4` on each machine — if it returns nothing, install Tailscale (`curl -fsSL https://tailscale.com/install.sh | sh`) and `tailscale up`.
2. On the head machine (`robros-MS-7E59` by default), run `tailscale ip -4`; that value is what `HEAD_IP` must be in [start_ray.sh](../start_ray.sh).
3. From a worker machine, `ping $HEAD_IP` to verify peering. If it fails, both machines need to be in the same tailnet (`tailscale status`).
4. After a successful `ray start`, run `ray status` — you should see 3 nodes with the resource counts `labeling_pc: 4`, `training_pc: 3`, `inference_pc: 1`.

If you redeployed on different hostnames, [start_ray.sh](../start_ray.sh)'s `case "$HOSTNAME"` block won't match — see [03_distributed_setup.md](03_distributed_setup.md#what-to-change-for-a-new-deployment).

### Too many open files / ulimit errors

**Symptom.** Training crashes with `OSError: [Errno 24] Too many open files`. Usually on the training machine after a few iterations.

**Where to look.** [start_ray.sh](../start_ray.sh) line 24 — `ulimit -n 65535` is only set on the `robros-ai1` branch. The trainer holds many memmap files and DataLoader worker fds open simultaneously.

**Fix.** Make sure `ulimit -n 65535` is in the branch for whatever your training machine's hostname is. Re-run `start_ray.sh`. To verify in the running shell: `ulimit -n` should report `65535` before you start Ray.

---

## Install and import

### ImportError or weights not loading: transformers patch

**Symptom.** `ImportError` or `AttributeError` from `transformers.models.*` when loading either OpenPI-based policy. Or the model loads but the first forward pass crashes complaining about a missing layer type.

**Where to look.** [openpi_transformer_lib_patch.sh](../openpi_transformer_lib_patch.sh) — this script copies OpenPI-specific replacement files into `.venv/lib/python3.12/site-packages/transformers/`. If you upgraded `transformers` or rebuilt the venv, the patch needs to run again.

**Fix.**

```bash
bash openpi_transformer_lib_patch.sh
```

The patch assumes `transformers==4.53.2` (the version [env_setup.sh](../env_setup.sh) installs). If you have a different version, the patch will technically still copy files but the model code may have moved underneath them.

### Submodule missing

**Symptom.** `ModuleNotFoundError: No module named 'trainer'` or `No module named 'robometer'`.

**Where to look.** [.gitmodules](../.gitmodules) — both `trainer/` and `data_labeler/auto/models/robometer/` are git submodules. A plain `git clone` without `--recurse-submodules` leaves them empty.

**Fix.**

```bash
git submodule update --init --recursive
```

For the Robometer one specifically, also re-run the editable install:

```bash
uv pip install -e ./data_labeler/auto/models/robometer
```

[data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py) also adds the submodule path to `sys.path` at import time, so even if the editable install gets lost, the actor imports succeed — but the editable install is still cleaner for development.

---

## First-run config errors

### norm_stats_file_path: File not found

**Symptom.** `File not found at: /home/robros/Projects/inference_engine/.../dataset_stats.pkl` printed on inference startup, followed by `TypeError: 'NoneType' object is not subscriptable` on the first inference.

**Where to look.** [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) — the `norm_stats_file_path` value is a hard-coded absolute path on the original developer's machine.

`read_stats_file()` in [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py) prints the warning and returns `None`. That `None` then propagates into `DataNormalizationInterface(robot=..., data_stats=None)` which crashes the first time anyone tries to `data_stats["observation.state"]`.

**Fix.** Edit the JSON to point at your local copy of `dataset_stats.pkl`. The file is produced by the trainer during dataset preparation — see [04_configuration.md](04_configuration.md#why-norm_stats_file_path-is-a-per-deployment-edit).

### --robot igris_c crashes immediately

**Symptom.** `python run_online_rl.py --robot igris_c` fails with `ModuleNotFoundError: No module named 'env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params'` before anything starts.

**Where to look.** [run_online_rl.py:92](../run_online_rl.py) — the `elif robot == "igris_c":` branch imports a module that doesn't exist. The `igris_c` runtime config directory only contains `init_params.py` and `__init__.py`; the `inference_runtime_params.py`, `inference_runtime_params.json`, and `inference_runtime_topics.json` are all missing.

**Fix.** Don't use `--robot igris_c` today. Use `igris_b`. If you need to add `igris_c` support, see [07_extending.md](07_extending.md#igris_c--what-is-already-there-what-is-missing).

### --train_config path doesn't exist

**Symptom.** Trainer crashes with `FileNotFoundError` on a path like `/home/user/Projects/online_rl/trainer/...`.

**Where to look.** [run_online_rl.py:179](../run_online_rl.py) — the default `--train_config` is an absolute path on the original developer's machine.

**Fix.** Pass `--train_config <your-path>` explicitly. Most training configs live under [trainer/experiment_training/reinforcement_learning/](../trainer/experiment_training/reinforcement_learning/). The trainer's per-experiment READMEs list what each YAML does.

---

## Runtime behaviour

### Replay buffer never produces samples / sample shape is wrong

**Symptom.** The trainer logs `replay buffer size: 0` over and over and never advances to `Train iter:`. Or it advances, but samples have unexpected shapes and the loss crashes.

**Where to look.**

- The trainer waits until `replay_buffer_size >= batch_size * 2 * world_size` before starting ([trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 428). With `batch_size` and `world_size=4`, that's at minimum 8 timesteps; in practice it needs many more — episodes must be fully labeled and added.
- For shape mismatches: [data_bridge/replay_buffer.py](../data_bridge/replay_buffer.py)'s `_build_offsets()` in `"lerobot_qchunk"` mode uses `obs_proprio_history` (default 50) and `action_horizon` (default 50). The window length is `max_offset - min_offset + 1`. The trainer's `data` schema must match the offsets — see [trainer/docs/03_ray_online_training.md](../trainer/docs/03_ray_online_training.md).

**Fix.**

1. Confirm the labeler is consuming the queue — check `ray status` for labeler actors, and grep logs for `[AutoRewardLabeler] Failed to process episode`.
2. Confirm `replay_buffer.add.remote(td)` is being called — every successful episode triggers it inside `process_episode`.
3. For shape issues, check that the trainer's expected offsets and the replay buffer's offsets agree. The replay buffer chunking parameters (`action_horizon=50`, `obs_proprio_history=50`, `obs_images_history=1`) are hardcoded in [data_bridge/replay_buffer.py](../data_bridge/replay_buffer.py)'s `ReplayBufferActor.__init__` defaults.

### Inference uses stale weights forever

**Symptom.** Trainer iterates and logs `Iteration: N -- Model: ... pushed from trainer`, but the inference loop never prints `Updating policy weights...`.

**Where to look.**

- [data_bridge/state_manager.py](../data_bridge/state_manager.py) — `get_state()` returns `None` when `controller_version == trainer_version`. If something else is pulling weights and bumping `controller_version`, the inference loop sees `None` forever.
- Weight loading happens only **between episodes** — [env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) calls `get_state.remote()` at the top of each outer `while True` iteration, just before signaling ready for the next episode. If your episodes are very long (>1000 steps default), updates won't show until episode N+1.
- The trainer only pushes every `save_every * 25` iterations — early in training, you won't see many pushes.
- **Named-actor coupling**: RTC calls `ray.get_actor("policy_state_manager")` by name. If you renamed the actor in [run_online_rl.py](../run_online_rl.py) but not in the inference loop, `ray.get_actor` raises `ValueError`.

**Fix.** Search for stray `policy_state_manager_handle.get_state.remote()` calls outside the inference loop. The named actor's name (`"policy_state_manager"`) must match in three places: [run_online_rl.py:80](../run_online_rl.py), [inference_loop.py:94](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py), and [trainer/trainer/online_trainer.py:424](../trainer/trainer/online_trainer.py).

The Sequential actor receives the handle as a constructor arg, so it's immune to rename mistakes but you should still keep all three consistent.

### GUI labeler shows blank frames

**Symptom.** The manual labeler window opens but the image area says "Waiting for video..." or shows garbled pixels.

**Where to look.** [data_labeler/human_in_the_loop/hil_reward_labeler.py](../data_labeler/human_in_the_loop/hil_reward_labeler.py) — `torch_frame_to_qimage()` handles both CHW and HWC layouts and both `uint8` and float `[0, 1]` ranges. If the frame is the wrong shape (e.g., a stacked image-history axis the GUI doesn't expect), it raises `ValueError` and gets caught in `_poll_for_work`, which then resets the UI.

**Fix.**

- "Waiting for video..." just means the queue is empty. That's expected if the env actor hasn't finished an episode yet.
- Garbled image: the frame is being interpreted as the wrong layout. The GUI expects `[T,H,W,3]` (HWC) or `[T,3,H,W]` (CHW). The episode recorder converts HWC→CHW before storing (see [env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py](../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py)). With CHW frames the GUI will permute correctly.
- "Reward tensor dtype must be signed or float to support -1" — the GUI's three reward buttons set −1/0/+1; if `reward` is `uint8` or `bool`, the negative one fails. The episode recorder initializes `reward` as `torch.zeros(1).squeeze()` (float32), so this should not happen unless your custom labeler changed the dtype.

### Why isn't my pedal/teleop doing anything when I pass --human_reward_labeler?

**Symptom.** You pass `--human_reward_labeler` expecting the foot pedal and teleop arms under [env_actor/human_in_the_loop/](../env_actor/human_in_the_loop/) to start working, and nothing happens.

**Where to look.** [run_online_rl.py:131-155](../run_online_rl.py).

`--human_reward_labeler` only switches which reward labeler runs:

| Flag | Reward labeler |
|---|---|
| off (default) | `AutoRewardLabelerActor` (Robometer VLM) |
| on | `ManualRewardLabelerActor` (PySide6 GUI) |

It does **not** wire in anything from [env_actor/human_in_the_loop/](../env_actor/human_in_the_loop/). That directory holds a parallel implementation of the inference path with teleoperation and pedal-based intervention — none of those files are imported by [run_online_rl.py](../run_online_rl.py). To activate them, a separate entrypoint would be needed (it does not exist in the repo today).

**Fix.** None — this is documented behaviour. If you actually want HIL teleop, you need to write the entrypoint. The scaffolding in [env_actor/human_in_the_loop/](../env_actor/human_in_the_loop/) is a starting template.

### Manual reward labeler never pumps the queue

**Symptom.** You pass `--human_reward_labeler`, see `running human labeler...` in the driver, the Qt window may open or may not — but the queue keeps filling and the trainer's `replay buffer size` never increases past 0.

**Where to look.** [run_online_rl.py:131-141](../run_online_rl.py). The manual branch creates the actor handle but **never calls** `labeler.start.remote()`. Only the auto branch calls `.start.remote()` (inside the `for i in range(num_labeler_gpus):` loop). `start()` is the method that begins the queue-polling loop and launches the Qt event loop, so without that call the actor sits idle.

**Fix.** This is a known issue. To unblock yourself today: use `--human_reward_labeler` only if you are also willing to patch [run_online_rl.py](../run_online_rl.py) to add `labeler.start.remote()` after the manual labeler is constructed. (Documentation-only PRs do not touch code; this note is here so a junior who patches it knows why.)

### Auto labeler cannot find task

**Symptom.** Auto labeler actor logs `[AutoRewardLabeler] Failed to process episode: ` followed by `ValueError` (no message). Reward never gets written.

**Where to look.** [data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py) line 78–83 — `process_episode` pulls `task = episode_data["task"]`; if the key is missing, it raises `ValueError`.

The task field has to be present in the TensorDict that the episode recorder produces. The current [EpisodeRecorderBridge](../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py) does **not** add a `"task"` field — it adds `"task_index"` only. If you are using the auto labeler today, either:

1. Modify the episode recorder to include a `"task"` string per timestep, or
2. Override the recorder to attach the task at episode-stack time.

This is a real coupling gap — the labeler expects a string, the recorder produces an integer index. The auto labeler is otherwise correct.

### Episode queue full, env actor stalls

**Symptom.** Env actor blocks on `episode_queue_handle.put(...)` for minutes. No new episodes are produced.

**Where to look.** [run_online_rl.py:78](../run_online_rl.py) — the queue has `maxsize=15`. If the labeler is too slow (or, with manual labeling, the human is too slow), `put(..., block=True)` blocks until the labeler dequeues.

**Fix.**

- Confirm the labeler is alive: `ray status` should show its actor; `ray logs` should show recent `[AutoRewardLabeler] Failed to process episode` or successful processing prints.
- Bump `--num_labeler_gpus` to add parallel labeler workers (auto only). The labeling machine needs that many GPUs.
- For manual labeling at 20 Hz × 1000-step episodes, expect a human to take significantly longer to label than the env actor takes to produce — the queue WILL back up. That's fine; the env actor blocks, the system stays consistent.

### Policy weights silently never update

**Symptom.** Like "Inference uses stale weights forever" but more subtle — the inference loop *does* print `received weight from plasma...` but the robot's behaviour never seems to change.

**Where to look.** [trainer/trainer/online_trainer.py](../trainer/trainer/online_trainer.py) line 536–540 — the trainer only pushes weights for models where the YAML's `component_build_args[model_name]['online_update']` is `true` and `freeze` is `false`. If you have a component that you want updated but the YAML has either of those wrong, the trainer silently filters it out.

The inference loop further filters: `if model_name in policy.components.keys()`. If the trainer's model name doesn't match a key in `policy.components`, the weight push is silently dropped.

**Fix.** Print `policy.components.keys()` on inference startup and `trainer.models.keys()` on training startup. They must overlap on the set of components you want updated online. The keys come from the YAMLs (`model.component_config_paths` on the env actor side, `model.component_build_args` on the trainer side) — keep them aligned.

---

If the symptom isn't here, [10_glossary.md](10_glossary.md) defines the vocabulary, and [02_architecture.md](02_architecture.md) shows the actor graph. Searching the codebase for the actor name (e.g. `StateManagerActor`) is usually the fastest path to the relevant code.
