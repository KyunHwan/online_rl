import ray
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, SliceSampler

@ray.remote
class ReplayBufferActor:
    def __init__(self, slice_len: int, capacity=10_000_000, scratch_dir="tmp/online_rl_data"):
        # LazyMemmapStorage is the key. It maps data to disk instantly.
        self.storage = LazyMemmapStorage(
            max_size=capacity, 
            scratch_dir=scratch_dir,
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

    @property
    def size(self):
        return len(self.buffer)

    def add(self, episode_tensordict):
        """
        Receives a TensorDict (living in Ray Shared RAM), 
        writes it to Disk, and releases the RAM reference.
        """
        # .extend() writes to the memmap file on disk
        self.buffer.extend(episode_tensordict)
        
        # Explicitly return True to signal completion
        return True

    def sample(self, batch_size):
        # Reads from Disk -> RAM for the trainer
        batch = self.buffer.sample(batch_size * self.episode_slice_len)
        return batch.reshape(batch_size, self.episode_slice_len)
    