import os
import shutil

import ray
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, SliceSampler


@ray.remote
class ResfitReplayBufferActor:
    def __init__(
        self,
        capacity=10_000_000,
        use_hil_buffer: bool = False,
        # keys as stored by EpisodeRecorderBridge
        proprio_key: str = "proprio",
        reward_key: str = "reward",
        action_key: str = "action",
        image_keys=("head", "left", "right"),

        # --- chunking params (resfit-style) ---
        action_horizon: int = 50,
        reward_horizon: int = 1,
        obs_subsample_step: int = 3,

        # SliceSampler settings
        strict_length: bool = True,
        compile: bool = False,
    ):
        self.proprio_key = proprio_key
        self.reward_key = reward_key
        self.action_key = action_key
        self.image_keys = image_keys

        self.use_hil_buffer = use_hil_buffer

        # ---- 1) Build integer step offsets (relative to anchor timestep) ----
        self.action_offsets, self.reward_offsets, self.proprio_offsets, self.image_offsets = \
            self._build_offsets(
                action_horizon=action_horizon,
                reward_horizon=reward_horizon,
                obs_subsample_step=obs_subsample_step,
            )

        # ---- 2) Compute window length + anchor index ----
        self.anchor, self.episode_slice_len = self._compute_window(self.action_offsets,
                                                                   self.reward_offsets,
                                                                   self.proprio_offsets,
                                                                   self.image_offsets)

        # ---- Storage (clean stale memmap files from previous runs) ----
        for d in ["/tmp/online_rl_auto_data", "/tmp/online_rl_hil_data"]:
            if os.path.exists(d):
                shutil.rmtree(d)

        self.storage = LazyMemmapStorage(
            max_size=capacity,
            scratch_dir="/tmp/online_rl_auto_data",
            existsok=True,
        )

        self.sampler = SliceSampler(
            slice_len=self.episode_slice_len,
            end_key=("next", "done"),
            strict_length=strict_length,
            compile=compile,
        )

        self.buffer = TensorDictReplayBuffer(storage=self.storage, sampler=self.sampler)

        if self.use_hil_buffer:
            self.hil_storage = LazyMemmapStorage(
                max_size=capacity,
                scratch_dir="/tmp/online_rl_hil_data",
                existsok=True,
            )
            self.hil_buffer = TensorDictReplayBuffer(storage=self.hil_storage, sampler=self.sampler)

    # --------------------------
    # offset + window utilities
    # --------------------------
    def _build_offsets(self, action_horizon, reward_horizon, obs_subsample_step):
        # Actions: from 0 (current) up to (horizon-1) steps in the future
        action_offsets = torch.arange(0, action_horizon, dtype=torch.long)

        # Rewards: separate horizon
        reward_offsets = torch.arange(0, reward_horizon, dtype=torch.long)

        # Observations: subsampled within action horizon, stepping by obs_subsample_step
        # e.g., action_horizon=50, step=3 -> [49, 46, 43, ..., 1]
        proprio_offsets = torch.arange(action_horizon - 1, -1, -obs_subsample_step, dtype=torch.long)

        # Images: same subsampling pattern as proprio
        image_offsets = torch.arange(action_horizon - 1, -1, -obs_subsample_step, dtype=torch.long)

        return action_offsets, reward_offsets, proprio_offsets, image_offsets

    def _compute_window(self, *offset_tensors: torch.Tensor):
        all_offs = torch.cat([t.reshape(-1) for t in offset_tensors if t.numel() > 0], dim=0)
        min_off = int(all_offs.min().item())
        max_off = int(all_offs.max().item())
        anchor = -min_off
        window_len = max_off - min_off + 1
        return anchor, window_len

    def _gather_time(self, window: TensorDict, key, offsets: torch.Tensor):
        """
        window: TensorDict with batch [B, T]
        returns: tensor with batch [B, len(offsets), ...]
        """
        idx = (self.anchor + offsets).to(torch.long)   # indices into time-dim
        x = window.get(key)                            # shape [B, T, ...]
        return x[:, idx]

    # --------------------------
    # public API
    # --------------------------
    def size(self):
        return len(self.buffer) + (len(self.hil_buffer) if self.use_hil_buffer else 0)

    def add(self, episode_tensordict, separate_key: str = "control_mode"):
        if not self.use_hil_buffer:
            self.buffer.extend(episode_tensordict)
        else:
            if episode_tensordict[separate_key][0].item() == 0:
                self.buffer.extend(episode_tensordict)
            else:
                self.hil_buffer.extend(episode_tensordict)
        return True

    def _sample_windows(self, rb: TensorDictReplayBuffer, num_windows: int) -> TensorDict:
        flat = rb.sample(num_windows * self.episode_slice_len)
        return flat.reshape(num_windows, self.episode_slice_len)

    def _pack_lerobot_like(self, window: TensorDict) -> TensorDict:
        """
        Convert [B, T] window -> [B] samples with chunked fields.
        Keys match resfit_lerobot_data.py delta_timestamps structure.
        """
        B = window.batch_size[0]

        out = TensorDict({}, batch_size=[B], device=None)

        # Anchor-step metadata (scalar per sample)
        if "episode" in window.keys():
            out["episode"] = window["episode"][:, self.anchor].clone()
        if "control_mode" in window.keys():
            out["control_mode"] = window["control_mode"][:, self.anchor].clone()
        if "task_index" in window.keys():
            out["task_index"] = window["task_index"][:, self.anchor].squeeze(-1).clone()

        # Action / reward horizon
        out["action"] = self._gather_time(window, self.action_key, self.action_offsets).clone()
        out["labels.reward"] = self._gather_time(window, self.reward_key, self.reward_offsets).clone()

        if "base_policy_action" in window.keys():
            out["base_policy_action"] = self._gather_time(window, "base_policy_action", self.action_offsets).clone()

        # Proprio history/chunk — both observation.current and observation.state
        proprio_data = self._gather_time(window, self.proprio_key, self.proprio_offsets).clone()
        out["observation.current"] = proprio_data
        out["observation.state"] = proprio_data

        # Done flags aligned with reward horizon
        if ("next", "done") in window.keys(True):
            out["labels.done"] = self._gather_time(window, ("next", "done"), self.reward_offsets).clone()

        # Images at subsampled offsets (same pattern as proprio)
        for k in self.image_keys:
            if k in window.keys():
                cam_name = {
                    "head": "cam_head",
                    "left": "cam_left",
                    "right": "cam_right",
                }.get(k, k)

                out[f"observation.images.{cam_name}"] = self._gather_time(window, k, self.image_offsets).clone()

        return out

    def _sample_and_pack(self, rb: TensorDictReplayBuffer, num_windows: int) -> TensorDict:
        windows = self._sample_windows(rb, num_windows)    # [B, T]
        return self._pack_lerobot_like(windows)            # [B]

    def sample(self, batch_size: int) -> TensorDict:
        """
        Returns batch_size independent LeRobot-style samples (each one contains its own
        obs-history and action-horizon chunks).
        """
        if not self.use_hil_buffer:
            return self._sample_and_pack(self.buffer, batch_size)

        bs = batch_size // 2
        a = self._sample_and_pack(self.buffer, bs)
        b = self._sample_and_pack(self.hil_buffer, bs)
        return TensorDict.cat([a, b], dim=0)
