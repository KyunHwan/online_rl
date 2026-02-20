import numpy as np

class DataNormalizationInterface:
    def __init__(self, 
                 robot,
                 data_stats):
        if robot == 'igris_b':
            from .robots.igris_b.data_normalization_manager import DataNormalizationBridge
        elif robot == 'igris_c': 
            from .robots.igris_c.data_normalization_manager import DataNormalizationBridge
        
        self.data_normalizer = DataNormalizationBridge(data_stats)
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.data_normalizer.normalize_state(state)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return self.data_normalizer.normalize_action(action)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return self.data_normalizer.denormalize_action(action)