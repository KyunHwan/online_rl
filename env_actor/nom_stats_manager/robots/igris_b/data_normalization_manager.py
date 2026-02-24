import numpy as np


class DataNormalizationBridge:
    def __init__(self, norm_stats):
        self.norm_stats = norm_stats
    
    def normalize_state(self, state: dict[str, np.ndarray]):
        # Load normalization stats
        state_mean = np.concatenate([self.norm_stats['observation.state']['mean'],
                                 self.norm_stats['observation.current']['mean']], axis=-1)
        state_std = np.concatenate([self.norm_stats['observation.state']['std'],
                                    self.norm_stats['observation.current']['std']], axis=-1)

        eps = 1e-8


        # Normalize proprioceptive state
        proprio_len = state['proprio'].shape[-1]
        state['proprio'] = (state['proprio'] - state_mean[:proprio_len]) / (state_std[:proprio_len] + eps)

        # cams
        for key in state.keys():
            if key != 'proprio':
                state[key] = state[key] / 255.0

        return state
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = self.norm_stats['action']['mean']
        action_std = self.norm_stats['action']['std']

        eps = 1e-8

        action = ((action - action_mean) / (action_std + eps))
        pass
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize action using stats.

        Args:
            action: Normalized action tensor

        Returns:
            Denormalized action as numpy array
        """
        action_mean = self.norm_stats['action']['mean']
        action_std = self.norm_stats['action']['std']

        denormalized = action * action_std + action_mean
        return denormalized


