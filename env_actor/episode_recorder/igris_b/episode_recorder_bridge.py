import tensordict
import numpy as np

class EpisodeRecorderBridge:
    def __init__(self):
        self.episodic_obs_state = []
        self.episodic_action = []

    def serve_train_data_buffer(self, episode_id):
        # 
        return
    
    def add_obs_state(self, obs_data: dict[str, np.array]):
        self.episodic_obs_state.append(obs_data)
    
    def add_action(self, action: np.array):
        self.episodic_action.append(action)
