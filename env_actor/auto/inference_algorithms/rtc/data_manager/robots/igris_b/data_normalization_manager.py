import torch
import numpy as np


class DataNormalizationBridge:
    def __init__(self, norm_stats):
        self.norm_stats = norm_stats

    def normalize_state_action(self, state_action_data, device):
        """
        state_action_data holds arrays in torch.Tensor format
        state_action_data has 
            action
            proprio
            head
            left 
            right
        """
        # Load normalization stats
        state_mean = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['mean']),
                                   torch.from_numpy(self.norm_stats['observation.current']['mean'])], dim=-1).to(device)
        state_std = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['std']),
                                  torch.from_numpy(self.norm_stats['observation.current']['std'])], dim=-1).to(device)
        
        action_mean = torch.from_numpy(self.norm_stats['action']['mean']).to(device)
        action_std = torch.from_numpy(self.norm_stats['action']['std']).to(device)

        eps = 1e-8

        # Normalize proprioceptive state
        # Expand batch shape
        state_action_data['proprio'] = ((torch.from_numpy(state_action_data['proprio']) - state_mean) / (state_std + eps)).unsqueeze(0)
        state_action_data['action'] = ((torch.from_numpy(state_action_data['action']) - action_mean) / (action_std + eps)).unsqueeze(0)

        for key in state_action_data.keys():
            if isinstance(state_action_data[key], np.ndarray):
                state_action_data[key] = torch.from_numpy(state_action_data[key]).to(device)

        return state_action_data
    
    def normalize_state(self, state_data, device):
        # Load normalization stats
        state_mean = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['mean']),
                                   torch.from_numpy(self.norm_stats['observation.current']['mean'])], dim=-1).to(device)
        state_std = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['std']),
                                  torch.from_numpy(self.norm_stats['observation.current']['std'])], dim=-1).to(device)

        eps = 1e-8

        # Normalize proprioceptive state
        # Expand batch shape
        state_data['proprio'] = ((torch.from_numpy(state_data['proprio']) - state_mean) / (state_std + eps)).unsqueeze(0)

        for key in state_data.keys():
            if isinstance(state_data[key], np.ndarray):
                state_data[key] = torch.from_numpy(state_data[key]).to(device)

        return state_data
    
    def normalize_action(self, action: torch.Tensor) -> np.ndarray:
        pass
    
    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
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
    
    def get_running_stats_with(self, episodic_stats):
        return None

