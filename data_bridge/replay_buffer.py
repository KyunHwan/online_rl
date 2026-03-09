import ray
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, SliceSampler


@ray.remote
class ReplayBufferActor:
    def __init__(
        self,
        capacity=10_000_000,
        use_hil_buffer: bool = False,
        # keys as stored by EpisodeRecorderBridge
        proprio_key: str = "proprio",
        reward_key: str = "reward",
        action_key: str = "action",
        image_keys=("head", "left", "right"),

        # --- chunking params (LeRobot-like) ---
        action_horizon: int = 50,
        obs_proprio_history: int = 50,
        obs_images_history: int = 1,
        chunking_mode: str = "lerobot_qchunk",   # "classic" | "lerobot_qchunk"

        # SliceSampler settings
        strict_length: bool = True,
        compile: bool = True,
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
                obs_proprio_history=obs_proprio_history,
                obs_images_history=obs_images_history,
                mode=chunking_mode,
            )

        # ---- 2) Compute window length + anchor index ----
        self.anchor, self.episode_slice_len = self._compute_window(self.action_offsets,
                                                                   self.reward_offsets,
                                                                   self.proprio_offsets,
                                                                   self.image_offsets)

        # ---- Storage ----
        self.storage = LazyMemmapStorage(
            max_size=capacity,
            scratch_dir="tmp/online_rl_auto_data",
        )

        # SliceSampler: batch-size must be divisible by slice_len. :contentReference[oaicite:1]{index=1}
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
                scratch_dir="tmp/online_rl_hil_data",
            )
            self.hil_buffer = TensorDictReplayBuffer(storage=self.hil_storage, sampler=self.sampler)

    # --------------------------
    # offset + window utilities
    # --------------------------
    def _build_offsets(self, action_horizon, obs_proprio_history, obs_images_history, mode: str):
        if mode == "classic":
            # Actions: future horizon
            action_offsets = torch.arange(0, action_horizon, dtype=torch.long)
            reward_offsets = torch.arange(0, action_horizon, dtype=torch.long)

            # Proprio: past history (including current)
            # e.g., H=3 -> [-2, -1, 0]
            proprio_offsets = torch.arange(-(obs_proprio_history - 1), 1, dtype=torch.long)

            # Images: past history (including current)
            image_offsets = torch.arange(-(obs_images_history - 1), 1, dtype=torch.long)

        elif mode == "lerobot_qchunk":
            # Match the integer ranges in your factory (since timestamps = offset * dt)
            action_offsets = torch.arange(0, action_horizon, dtype=torch.long)
            reward_offsets = torch.arange(0, action_horizon, dtype=torch.long)

            # e.g., H=2 -> [2, 1, 0, -1]
            proprio_offsets = torch.arange(obs_proprio_history, -obs_proprio_history, -1, dtype=torch.long)

            # e.g., H=2 -> [2, 0]  (this matches your exact code)
            # If you *meant* to use obs_images_history, replace this with something based on that.
            image_offsets = torch.arange(obs_proprio_history, -1, -obs_proprio_history, dtype=torch.long)

        else:
            raise ValueError(f"Unknown chunking_mode={mode}. Use 'classic' or 'lerobot_qchunk'.")

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
        # SliceSampler expects total items divisible by slice_len. :contentReference[oaicite:2]{index=2}
        flat = rb.sample(num_windows * self.episode_slice_len)
        # Some TorchRL versions already return [num_windows, T]; this reshape is safe either way.
        return flat.reshape(num_windows, self.episode_slice_len)

    def _pack_lerobot_like(self, window: TensorDict) -> TensorDict:
        """
        Convert [B, T] window -> [B] samples with chunked fields.
        Keys are returned in a LeRobot-ish naming style so your trainer can be shared.
        """
        B = window.batch_size[0]

        out = TensorDict({}, batch_size=[B], device=None)

        # Anchor-step metadata (scalar per sample)
        if "episode" in window.keys():
            out["episode"] = window["episode"][:, self.anchor].clone()
        if "control_mode" in window.keys():
            out["control_mode"] = window["control_mode"][:, self.anchor].clone()
        if "task_index" in window.keys():
            out["task_index"] = window["task_index"][:, self.anchor].clone()

        # Action / reward horizon
        out["action"] = self._gather_time(window, self.action_key, self.action_offsets).clone()
        out["labels.reward"] = self._gather_time(window, self.reward_key, self.reward_offsets).clone()

        # Proprio history/chunk
        out["observation.state"] = self._gather_time(window, self.proprio_key, self.proprio_offsets).clone()

        # Done flags aligned with reward horizon (optional but often useful)
        if ("next", "done") in window.keys(True):
            out["labels.done"] = self._gather_time(window, ("next", "done"), self.reward_offsets).clone()

        # Images at desired offsets (only materialize those frames)
        for k in self.image_keys:
            if k in window.keys():
                # Map your stored keys -> LeRobot-ish camera names if you want
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

# @ray.remote
# class ReplayBufferActor:
#     def __init__(self, slice_len: int, capacity=10_000_000, use_hil_buffer: bool=False):
#         # LazyMemmapStorage is the key. It maps data to disk instantly.
#         self.storage = LazyMemmapStorage(
#             max_size=capacity, 
#             scratch_dir="tmp/online_rl_auto_data",
#         )

#         self.episode_slice_len = slice_len
#         self.sampler = SliceSampler(
#             slice_len=self.episode_slice_len,
#             #traj_key="episode",
#             end_key=("next", "done"), # default in SliceSampler; set to your "done" location
#             #truncated_key=("next", "_slice_truncated"),
#             strict_length=True, # drop episodes shorter than SLICE_LEN
#             compile=True
#         )

#         self.buffer = TensorDictReplayBuffer(
#             storage=self.storage,
#             sampler=self.sampler
#         )

#         self.use_hil_buffer = use_hil_buffer
#         if self.use_hil_buffer:
#             self.hil_storage = LazyMemmapStorage(
#                 max_size=capacity, 
#                 scratch_dir="tmp/online_rl_hil_data",
#             )
#             self.hil_buffer = TensorDictReplayBuffer(
#                 storage=self.hil_storage,
#                 sampler=self.sampler
#             )
            
#     def size(self):
#         if self.use_hil_buffer:
#             return len(self.buffer) + len(self.hil_buffer)
#         else:
#             return len(self.buffer)

#     def add(self, episode_tensordict, separate_key: str = 'control_mode'):
#         """
#         Receives a TensorDict (living in Ray Shared RAM), 
#         writes it to Disk, and releases the RAM reference.
#         """
#         # .extend() writes to the memmap file on disk
#         if not self.use_hil_buffer:
#             self.buffer.extend(episode_tensordict)
#         else:
#             if episode_tensordict[separate_key][0].item() == 0:
#                 self.buffer.extend(episode_tensordict)
#             else:
#                 self.hil_buffer.extend(episode_tensordict)
        
#         # Explicitly return True to signal completion
#         return True

#     def sample(self, batch_size):
#         # Reads from Disk -> RAM for the trainer
#         if not self.use_hil_buffer:
#             batch = self.buffer.sample(batch_size * self.episode_slice_len)
#             return batch.reshape(batch_size, self.episode_slice_len)
#         else:
#             bs = batch_size // 2
#             batch = self.buffer.sample(bs * self.episode_slice_len).reshape(bs, self.episode_slice_len)
#             hil_batch = self.hil_buffer.sample(bs * self.episode_slice_len).reshape(bs, self.episode_slice_len)
#             return torch.cat([batch, hil_batch], dim=0)

        
    