import ray
import torch

@ray.remote(num_gpus=1)
class RewardLabelerActor:
    def __init__(self, episode_queue_handle, replay_buffer_handle):
        self.episode_queue_handle = episode_queue_handle
        self.replay_buffer_handle = replay_buffer_handle
        # Load your heavy VLM here
        # self.vlm = Cosmos2...

    def start(self):
        while True:
            # Wait until there's data
            # Resolve the object (Zero-copy read from Plasma) internally
            episode_data = self.episode_queue_handle.get(block=True)
            
            # Label Rewards & Send labeled data
            self.process_episode(episode_data)

    def process_episode(self, episode_data):
        # Label rewards
        # Assume we modify the tensordict or create a new one with rewards
        # TODO: 
        #     episode_data["reward"] = reward
        
        # Push to Replay Buffer
        # We call .remote() and wait for the acknowledgment to ensure 
        # the disk write is queued before we declare victory.
        replay_buffer_write_ref = self.replay_buffer_handle.add.remote(episode_data)
        
        # Wait for the disk write to register so we know flow is moving
        data_added = ray.get(replay_buffer_write_ref)
