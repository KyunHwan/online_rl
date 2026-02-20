import torch
import numpy as np


class DataNormalizationBridge:
    def __init__(self, norm_stats):
        self.norm_stats = norm_stats
    
    def normalize_state(self, state):
        # Load normalization stats
        state_mean = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['mean']),
                                   torch.from_numpy(self.norm_stats['observation.current']['mean'])], dim=-1)
        state_std = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['std']),
                                  torch.from_numpy(self.norm_stats['observation.current']['std'])], dim=-1)

        eps = 1e-8

        # Normalize proprioceptive state
        # Expand batch shape
        state['proprio'] = ((torch.from_numpy(state['proprio']) - state_mean) / (state_std + eps)).unsqueeze(0)

        for key in state.keys():
            if isinstance(state[key], np.ndarray):
                state[key] = torch.from_numpy(state[key])

        return state
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = torch.from_numpy(self.norm_stats['action']['mean'])
        action_std = torch.from_numpy(self.norm_stats['action']['std'])

        eps = 1e-8

        action['action'] = ((torch.from_numpy(action['action']) - action_mean) / (action_std + eps)).unsqueeze(0)
        pass
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize action using stats.

        Args:
            action: Normalized action tensor

        Returns:
            Denormalized action as numpy array
        """
        device = action.device
        action_mean = torch.from_numpy(self.norm_stats['action']['mean']).to(device)
        action_std = torch.from_numpy(self.norm_stats['action']['std']).to(device)

        # Denormalize: action = action * std + mean
        denormalized = action * action_std + action_mean
        return denormalized.cpu().numpy()


