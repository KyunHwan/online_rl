import ray
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, SliceSampler

@ray.remote
class ReplayBufferActor:
    def __init__(self, slice_len: int, capacity=10_000_000, use_hil_buffer: bool=False):
        # LazyMemmapStorage is the key. It maps data to disk instantly.
        self.storage = LazyMemmapStorage(
            max_size=capacity, 
            scratch_dir="tmp/online_rl_auto_data",
        )

        self.episode_slice_len = slice_len
        self.sampler = SliceSampler(
            slice_len=self.episode_slice_len,
            traj_key="episode",
            #end_key=("next", "done"), # default in SliceSampler; set to your "done" location
            #truncated_key=("next", "_slice_truncated"),
            strict_length=True, # drop episodes shorter than SLICE_LEN
            compile=True
        )

        self.buffer = TensorDictReplayBuffer(
            storage=self.storage,
            sampler=self.sampler
        )

        self.use_hil_buffer = use_hil_buffer
        if self.use_hil_buffer:
            self.hil_storage = LazyMemmapStorage(
                max_size=capacity, 
                scratch_dir="tmp/online_rl_hil_data",
            )
            self.hil_buffer = TensorDictReplayBuffer(
                storage=self.hil_storage,
                sampler=self.sampler
            )
            
    @property
    def size(self):
        return len(self.buffer)

    def add(self, episode_tensordict, separate_key: str = 'control_mode'):
        """
        Receives a TensorDict (living in Ray Shared RAM), 
        writes it to Disk, and releases the RAM reference.
        """
        # .extend() writes to the memmap file on disk
        if not self.use_hil_buffer:
            self.buffer.extend(episode_tensordict)
        else:
            if episode_tensordict[separate_key][0].item() == 0:
                self.buffer.extend(episode_tensordict)
            else:
                self.hil_buffer.extend(episode_tensordict)
        
        # Explicitly return True to signal completion
        return True

    def sample(self, batch_size):
        # Reads from Disk -> RAM for the trainer
        if not self.use_hil_buffer:
            batch = self.buffer.sample(batch_size * self.episode_slice_len)
            return batch.reshape(batch_size, self.episode_slice_len)
        else:
            bs = batch_size // 2
            batch = self.buffer.sample(bs * self.episode_slice_len).reshape(bs, self.episode_slice_len)
            hil_batch = self.hil_buffer.sample(bs * self.episode_slice_len).reshape(bs, self.episode_slice_len)
            return torch.cat([batch, hil_batch], dim=0)

        
    