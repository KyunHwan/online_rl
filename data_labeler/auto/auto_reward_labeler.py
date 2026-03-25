import ray
import torch
import numpy as np
import traceback
import sys
import pathlib
from typing import Optional, Tuple

# Ensure robometer submodule is importable (data_labeler/auto/models/robometer/)
_robometer_root = str(pathlib.Path(__file__).resolve().parent / "models" / "robometer")
if _robometer_root not in sys.path:
    sys.path.insert(0, _robometer_root)

@ray.remote(num_gpus=1)
class AutoRewardLabelerActor:
    def __init__(self, 
                 episode_queue_handle, 
                 replay_buffer_actor, 
                 img_frame_key: str, 
                 reward_key: str,
                 num_subsampled_frames: int=32,
                 model_path: str = "robometer/Robometer-4B"):
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        self.episode_queue_handle = episode_queue_handle
        self.replay_buffer_actor = replay_buffer_actor

        self.num_subsampled_frames = num_subsampled_frames
        self.img_frame_key = img_frame_key
        self.reward_key = reward_key
        self.device = torch.device("cuda")

        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        config, tokenizer, processor, model = load_model_from_hf(
            model_path=model_path, device=self.device
        )
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.batch_collator = setup_batch_collator(processor, tokenizer, config, is_eval=True)

        loss_config = getattr(config, "loss", None)
        self.is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config else False
        )
        self.num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(config.model, "progress_discrete_bins", 10)
        )

    def start(self):
        while True:
            episode_data = self.episode_queue_handle.get(block=True)
            try:
                self.process_episode(episode_data)
            except Exception as e:
                print(f"[AutoRewardLabeler] Failed to process episode: {e}")
                traceback.print_exc()

    def process_episode(self, episode_data):
        from robometer.data.dataset_types import Trajectory, ProgressSample
        from robometer.evals.eval_server import compute_batch_outputs

        # CHW uint8 (T, C, H, W) -> HWC uint8 (T, H, W, 3)
        full_frames_chw = episode_data[self.img_frame_key]
        full_frames_hwc = full_frames_chw.permute(0, 2, 3, 1).contiguous().numpy()
        T = full_frames_hwc.shape[0]

        frames_hwc, indices = self.subsample_frames(full_frames_hwc, self.num_subsampled_frames)

        # Get task description from tensordict, fallback to task_index lookup
        if "task" in episode_data.keys():
            task = episode_data["task"]
            if not isinstance(task, str):
                task = str(task)
        else:
            raise ValueError

        # Build robometer input
        traj = Trajectory(
            frames=frames_hwc,
            frames_shape=tuple(frames_hwc.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")

        # Collate, move to GPU, infer
        batch = self.batch_collator([progress_sample])
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)

        results = compute_batch_outputs(
            self.model, 
            self.tokenizer, 
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=self.is_discrete,
            num_bins=self.num_bins,
        )

        # Extract per-frame progress scores as rewards
        progress_pred = results.get("progress_pred", [])
        progress_scores = (
            np.array(progress_pred[0], dtype=np.float32)
            if progress_pred and len(progress_pred) > 0
            else np.array([], dtype=np.float32)
        )
        
        # Extract per-frame success probabilities
        outputs_success = results.get("outputs_success", {})
        success_probs_list = outputs_success.get("success_probs", []) if outputs_success else []
        success_scores = (
            np.array(success_probs_list[0], dtype=np.float32)
            if success_probs_list and len(success_probs_list) > 0
            else np.array([], dtype=np.float32)
        )

        progress_scores = self.interpolate_to_full(progress_scores, indices, T)
        success_scores = self.interpolate_to_full(success_scores, indices, T)

        episode_data[self.reward_key] = torch.from_numpy(progress_scores)
        if success_scores.size > 0:
            episode_data["success_probs"] = torch.from_numpy(success_scores)

        # Push to replay buffer
        replay_buffer_write_ref = self.replay_buffer_actor.add.remote(episode_data)
        ray.get(replay_buffer_write_ref)
    
    def subsample_frames(self, frames: np.ndarray, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Uniformly subsample frames, always including first and last frame.

        Returns (subsampled_frames, indices) where indices are the selected frame positions.
        """
        T = frames.shape[0]
        if num_frames >= T:
            return frames, np.arange(T)
        indices = np.round(np.linspace(0, T - 1, num_frames)).astype(int)
        return frames[indices], indices
    
    def interpolate_to_full(self, values: np.ndarray, indices: np.ndarray, total_frames: int) -> np.ndarray:
        """Linearly interpolate subsampled values back to full frame count."""
        if values.size == 0:
            return values
        full_indices = np.arange(total_frames)
        return np.interp(full_indices, indices, values)
