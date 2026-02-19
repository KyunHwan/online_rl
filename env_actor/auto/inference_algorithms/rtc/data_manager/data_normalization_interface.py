import numpy as np
import torch

class DataNormalizationInterface:
    def __init__(self, 
                 robot,
                 data_stats):
        if robot == 'igris_b':
            from .robots.igris_b.data_normalization_manager import DataNormalizationBridge
        elif robot == 'igris_c': 
            from .robots.igris_c.data_normalization_manager import DataNormalizationBridge
        
        self.data_normalizer = DataNormalizationBridge(data_stats)
    
    def normalize_state_action(self, state_action_data, device):
        return self.data_normalizer.normalize_state_action(state_action_data, device)
    
    def normalize_state(self, state_data, device):
        return self.data_normalizer.normalize_state(state_data, device)

    def normalize_action(self, action, device):
        return self.data_normalizer.normalize_action(action, device)

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        return self.data_normalizer.denormalize_action(action)

    def get_running_stats_with(self, episodic_stats):
        return self.data_normalizer.get_running_stats_with(episodic_stats)