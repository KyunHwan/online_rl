import ray
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=10_000_000, scratch_dir="tmp/online_rl_data"):
        # LazyMemmapStorage is the key. It maps data to disk instantly.
        self.storage = LazyMemmapStorage(
            max_size=capacity, 
            scratch_dir=scratch_dir,
        )
        self.buffer = TensorDictReplayBuffer(
            storage=self.storage,
        )

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
        return self.buffer.sample(batch_size)