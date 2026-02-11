import numpy as np
import torch

class DataNormalizationInterface:
    def __init__(self, 
                 robot,
                 data_stats):
        if robot == 'igris_b':
            from .igris_b.data_normalization_manager import DataNormalizationBridge
        elif robot == 'igris_c': 
            from .igris_c.data_normalization_manager import DataNormalizationBridge
        
        self.data_normalizer = DataNormalizationBridge(data_stats)
    
    def normalize_state_action(self, state_action_data):
        return self.data_normalizer.normalize_state_action(state_action_data)

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        return self.data_normalizer.denormalize_action(action)

        
