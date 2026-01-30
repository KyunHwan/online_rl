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
            end_key=("next", "done"), # default in SliceSampler; set to your "done" location
            strict_length=True, # drop episodes shorter than SLICE_LEN
            compile=True
        )
        """
        # ---------------------------------------------------------------------
        # Example: extend with ONE trajectory TensorDict of shape [T]
        # (In practice you'll extend with data from your collector / dataset.)
        T = 1000
        traj = TensorDict(
            {
                "proprio": torch.randn(T, 16),
                "action": torch.randn(T, 7),
                ("next", "done"): torch.zeros(T, 1, dtype=torch.bool),
            },
            batch_size=[T],
        )
        traj[("next", "done")][-1] = True          # mark end of episode
        rb.extend(traj)
        # ---------------------------------------------------------------------
        """

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
    