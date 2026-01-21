import ray
from tensordict import TensorDict
import numpy as np
import torch




@ray.remote
class ControllerActor:
    def __init__(self, episode_queue_handle, inference_config_path, shared_memory_names):
        self.episode_queue_handle = episode_queue_handle
        self.inference_config_path = inference_config_path
    
    def start(self,):
        episodic_data = []
        while True:
            if len(episodic_data) != 0:
                # wait until there's room to put the data in the queue
                episodic_data_ref = ray.put(TensorDict.stack(episodic_data, dim=0))
                self.episode_queue_handle.put(episodic_data_ref,
                                              block=True)
            episodic_data = []
            for step in range(900):
                episodic_data.append(TensorDict({
                    'reward': torch.randn(40, 24),
                    'action': torch.ones(40,24),
                    'state': torch.zeros(40, 24)
                }, batch_size=[]))
                
