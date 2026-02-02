import tensordict
from tensordict import TensorDict
import numpy as np
import torch
import copy

class EpisodeRecorderBridge:
    def __init__(self):
        self.episodic_obs_state = []
        self.episodic_action = []

    def serve_train_data_buffer(self, episode_id):
        """
        Returns a tensordict for SliceSampler replay buffer
        """
        train_data = [TensorDict({
            "episode": torch.tensor(episode_id, dtype=torch.int64),
            ("next", "done"): torch.zeros(1, dtype=torch.bool)
        }) for i in range(len(self.episodic_obs_state))]
        train_data = torch.stack(train_data, dim=0)
        train_data[("next", "done")][-1] = True

        next_obs = copy.deepcopy(self.episodic_obs_state[1:])
        next_obs.append(next_obs[-1])
        next_obs = TensorDict({
            "next": torch.stack(next_obs, dim=0)
        })

        obs = torch.stack(self.episodic_obs_state, dim=0)
        actions = torch.stack(self.episodic_action, dim=0)

        train_data.update(obs)
        train_data.update(actions)
        train_data.update(next_obs)

        return train_data
    
    def add_obs_state(self, obs_data: dict[str, np.array]):
        """
        Fields of obs_data are:
            proprio
            head
            left
            right
        where head, left, right represent images.
        """
        obs_data_tensordict = TensorDict(
            {
                data_name: torch.from_numpy(obs_data[data_name]) for data_name in obs_data.keys()
            },
            batch_size=[]
        )
        self.episodic_obs_state.append(obs_data_tensordict)
    
    def add_action(self, action: np.array):
        action_tensordict = TensorDict({
            'action': torch.from_numpy(action)
        },
        batch_size=[])
        self.episodic_action.append(action_tensordict)
    
    def init_train_data_buffer(self):
        self.episodic_obs_state = []
        self.episodic_action = []
