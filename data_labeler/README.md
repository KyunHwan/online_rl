# data_labeler

Reward annotation for collected episodes. Two labelers are available, selected by the `--human_reward_labeler` flag in [run_online_rl.py](../run_online_rl.py):

1. **Automatic** (default) — [Robometer](../docs/10_glossary.md#robometer) labels rewards via a VLM.
2. **Manual** — a PySide6 GUI lets a human assign per-frame rewards.

Both actors implement the same Ray Queue → annotate → ReplayBuffer pipeline. See [../docs/02_architecture.md](../docs/02_architecture.md#actor-autorewardlabeleractor-or-manualrewardlabeleractor) for where they fit in the cluster.

## Table of contents

- [Automatic Reward Labeler (default)](#automatic-reward-labeler-default)
- [Manual Reward Labeler](#manual-reward-labeler)
- [Data flow](#data-flow)
- [Known issues](#known-issues)
- [Files](#files)

## Automatic Reward Labeler (default)

**File:** [`auto/auto_reward_labeler.py`](auto/auto_reward_labeler.py)

[`AutoRewardLabelerActor`](auto/auto_reward_labeler.py) is a Ray actor (`@ray.remote(num_gpus=1)`). It pulls episodes off the `episode_queue`, runs them through the [Robometer-4B](https://huggingface.co/robometer/Robometer-4B) vision-language reward model, and writes the per-frame progress score back into the TensorDict as the `reward` field.

### Constructor signature

```python
AutoRewardLabelerActor(
    episode_queue_handle,
    replay_buffer_actor,
    img_frame_key: str,                 # which camera key to feed the VLM, e.g. "head"
    reward_key: str,                    # which TD key to write the per-frame score into, e.g. "reward"
    num_subsampled_frames: int = 32,    # frames sampled per episode for VLM scoring
    model_path: str = "robometer/Robometer-4B",
)
```

[run_online_rl.py](../run_online_rl.py) spawns `--num_labeler_gpus` of these (default 4) and calls `.start.remote()` on each.

### How `process_episode` works

1. Pull a TensorDict off the queue (RayQueue auto-dereferences the `ObjectRef`).
2. Take the camera frames from `episode_data[img_frame_key]`. Permute CHW → HWC and convert to numpy uint8.
3. Read the task description from `episode_data["task"]`. If the key is missing, raise `ValueError`. **The task is not configurable via a constructor kwarg** — it must already be in the TensorDict.
4. Uniformly subsample 32 frames (always including first and last).
5. Run the Robometer batch collator and `compute_batch_outputs(..., sample_type="progress")`.
6. Interpolate the per-frame progress scores back to the full episode length with `np.interp`.
7. Write `episode_data["reward"] = torch.from_numpy(progress_scores)`. If `outputs_success` is present, also write `episode_data["success_probs"]`.
8. `ray.get(self.replay_buffer_actor.add.remote(episode_data))` — blocks on disk write so this actor doesn't out-pace the buffer.

### Setup

Robometer is a git submodule at `auto/models/robometer/`. The repo install runs `uv pip install -e ./data_labeler/auto/models/robometer` ([env_setup.sh](../env_setup.sh)). Additionally, [auto/auto_reward_labeler.py](auto/auto_reward_labeler.py) inserts the submodule path into `sys.path` at module import (lines 9–12), so the actor works even if the editable install was lost.

If you clone without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
uv pip install -e ./data_labeler/auto/models/robometer
```

## Manual Reward Labeler

**File:** [`human_in_the_loop/hil_reward_labeler.py`](human_in_the_loop/hil_reward_labeler.py)

[`ManualRewardLabelerActor`](human_in_the_loop/hil_reward_labeler.py) opens a PySide6 (Qt) window with a video slider and three reward buttons (−1, 0, +1) plus a Complete button. A `QTimer` polls the Ray Queue every 100 ms.

### Constructor signature

```python
ManualRewardLabelerActor(
    episode_queue_handle,
    replay_buffer_actor,
    img_frame_key: str = "head",
    reward_key: str = "reward",
    window_title: str = "Reward Labeler",
)
```

### GUI behaviour

`torch_frame_to_qimage()` handles:

- CHW ↔ HWC layout (whichever the TensorDict stored).
- float `[0, 1]` → uint8 `[0, 255]` rescaling.
- Contiguous memory layout for Qt (`frame.contiguous()` before `QImage(...)`).

The reward tensor must be signed or float — uint8/bool cannot hold −1. The episode recorder initializes reward as float32, so this is normally fine.

On "Complete", the modified TensorDict is sent to `replay_buffer_actor.add.remote(...)` and the UI resets to wait for the next item.

## Data flow

```text
EnvActor                          AutoRewardLabeler                  ReplayBuffer
   │                                    │                                 │
   │ ray.put(td) → ref                  │                                 │
   ├──── episode_queue.put(ref) ──────▶ │                                 │
   │                                    │ ray.get(ref) → TensorDict        │
   │                                    │ subsample 32 frames              │
   │                                    │ Robometer scoring                │
   │                                    │ interpolate to full T            │
   │                                    │ td["reward"] = scores            │
   │                                    │                                  │
   │                                    ├── replay_buffer.add.remote(td) ─▶│
   │                                    │ ray.get(...)  # block on write   │ memmap
```

## Known issues

These are real, documented behaviours of the code as it stands. Fixing them is out of scope for this README pass.

- **Manual labeler never pumps the queue.** In [run_online_rl.py](../run_online_rl.py) lines 133–141, `ManualRewardLabelerActor` is instantiated but `labeler.start.remote()` is **not** called (only the auto branch calls `.start.remote()` on each labeler). The result: `--human_reward_labeler` spawns an idle actor and episodes accumulate in the queue until the env actor blocks on the bounded `put`. See [../docs/09_troubleshooting.md](../docs/09_troubleshooting.md#manual-reward-labeler-never-pumps-the-queue).
- **Auto labeler requires `episode_data["task"]`.** The current [EpisodeRecorderBridge](../env_actor/episode_recorder/robots/igris_b/episode_recorder_bridge.py) adds `"task_index"` (an integer) but not `"task"` (a string). Episodes from this recorder will raise `ValueError` in `AutoRewardLabelerActor.process_episode`. See [../docs/09_troubleshooting.md](../docs/09_troubleshooting.md#auto-labeler-cannot-find-task).

## Files

| File | Purpose |
|---|---|
| [`auto/auto_reward_labeler.py`](auto/auto_reward_labeler.py) | Robometer-based automatic reward labeling. |
| [`human_in_the_loop/hil_reward_labeler.py`](human_in_the_loop/hil_reward_labeler.py) | PySide6 GUI for manual reward labeling. |
| `auto/models/robometer/` | Robometer git submodule. **Out of scope** — do not edit. |
