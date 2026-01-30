import ray
import torch

@ray.remote(num_gpus=1)
class SequentialActor:
    def __init__(self, 
                 policy_state_manager_handle, 
                 episode_queue_handle,
                 inference_config_path):
        self.policy_state_manager_handle = policy_state_manager_handle
        self.episode_queue_handle = episode_queue_handle
        self.inference_config_path = inference_config_path
        self.policy = None

    def start(self):
        while True:
            current_weights_ref = self.policy_state_manager_handle.get_weights.remote()
            current_weights = ray.get(current_weights_ref)
            if current_weights is not None:
                new_weights = current_weights # Zero-copy fetch
                print("weights updated: ", new_weights.keys())
                #self.policy.load_state_dict(new_weights)