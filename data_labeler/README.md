# data_labeler

Reward annotation for collected episodes. Two labeling modes are available, selected via the `--human_reward_labeler` flag in [`run_online_rl.py`](../run_online_rl.py):

1. **Automatic** (default) — a Vision-Language Model labels rewards.
2. **Manual** — a PySide6 GUI lets a human assign per-frame binary rewards.

Both actors share the same interface: pull episodes from the Ray Queue → annotate rewards → push labeled data to the replay buffer.

## Automatic Reward Labeler

**File:** [`auto/auto_reward_labeler.py`](auto/auto_reward_labeler.py)

`AutoRewardLabelerActor` is a Ray actor (`num_gpus=1`) that uses [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) (a VLM based on Qwen3VL) to automatically generate reward labels from episode image frames.

### How It Works

1. Blocks on `episode_queue_handle.get()` until an episode is available (zero-copy read from Ray Plasma).
2. Calls `process_episode()` to run the VLM and annotate rewards.
3. Pushes the labeled TensorDict to the `ReplayBufferActor` via `replay_buffer_actor.add.remote()`.
4. Waits for the disk write to complete before processing the next episode.

### Configuration

- Model: `nvidia/Cosmos-Reason2-8B` with `flash_attention_2`
- Precision: `bfloat16`
- Image resolution: 224x224 (via processor settings)

## Manual Reward Labeler

**File:** [`human_in_the_loop/hil_reward_labeler.py`](human_in_the_loop/hil_reward_labeler.py)

`ManualRewardLabelerActor` is a Ray actor that runs a PySide6 (Qt) GUI for frame-by-frame binary reward labeling.

### GUI Features

- **Video slider** — scrub through episode frames.
- **Reward buttons** — set reward to 0 or 1 for the current frame.
- **Complete button** — pushes the labeled episode to the replay buffer.
- **Auto-polling** — polls the Ray Queue every 100ms for new episodes.
- Supports both `uint8` and float tensors, both HWC and CHW image layouts.

### How It Works

1. A `QTimer` polls `episode_queue_handle.get_nowait()` for new episodes.
2. When an episode arrives, it extracts image frames and reward tensors from the TensorDict.
3. The user navigates frames with the slider and sets binary rewards.
4. On "Complete", the mutated TensorDict (with updated rewards) is pushed to the replay buffer.

### Frame Conversion

`torch_frame_to_qimage()` handles:
- CHW → HWC permutation
- Float `[0, 1]` → `uint8` `[0, 255]` conversion
- Contiguous memory layout for Qt compatibility

## Data Flow

```
EnvActor
   │
   ├─ episode_queue.put(ray.put(tensordict))
   │
   ▼
RewardLabeler (auto or manual)
   │
   ├─ label rewards on tensordict
   │
   ├─ replay_buffer_actor.add.remote(labeled_tensordict)
   │
   ▼
ReplayBufferActor (disk write)
```

## Files

| File | Purpose |
|------|---------|
| [`auto/auto_reward_labeler.py`](auto/auto_reward_labeler.py) | VLM-based automatic reward labeling |
| [`human_in_the_loop/hil_reward_labeler.py`](human_in_the_loop/hil_reward_labeler.py) | PySide6 GUI for manual reward labeling |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `auto/` | Automatic reward labeling via VLM |
| `human_in_the_loop/` | Manual reward labeling via Qt GUI |
