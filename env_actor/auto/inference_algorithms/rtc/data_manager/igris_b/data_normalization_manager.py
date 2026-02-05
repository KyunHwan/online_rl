import torch
import numpy as np


class DataNormalizationBridge:
    def __init__(self, norm_stats):
        self.norm_stats = norm_stats

    def normalize_state_action(self, state, action):
        norm_obs_state = {}

        # Load normalization stats
        state_mean = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['mean']),
                                   torch.from_numpy(self.norm_stats['observation.current']['mean'])], dim=-1)
        state_std = torch.concat([torch.from_numpy(self.norm_stats['observation.state']['std']),
                                  torch.from_numpy(self.norm_stats['observation.current']['std'])], dim=-1)

        eps = 1e-8

        # Normalize proprioceptive state
        rh = torch.from_numpy(self.robot_proprio_history).to(device, dtype=torch.float32, non_blocking=True)

        # Handle different shapes of normalization stats
        state_dim = self.runtime_params.proprio_state_dim
        num_robot_obs = self.runtime_params.proprio_history_size

        if state_mean.numel() == state_dim and state_std.numel() == state_dim:
            rh = (rh - state_mean.view(1, -1)) / (state_std.view(1, -1) + eps)
        else:
            # In case stats are flattened differently, repeat to match shape
            total_size = num_robot_obs * state_dim
            sm_rep = state_mean if state_mean.numel() == total_size else state_mean.repeat((total_size + state_mean.numel() - 1) // state_mean.numel())[:total_size]
            ss_rep = state_std if state_std.numel() == total_size else state_std.repeat((total_size + state_std.numel() - 1) // state_std.numel())[:total_size]
            rh = (rh - sm_rep.view(num_robot_obs, state_dim)) / (ss_rep.view(num_robot_obs, state_dim) + eps)

        norm_obs_state['proprio'] = rh.reshape(1, -1).view(1, num_robot_obs, state_dim)
        
        # Normalize images
        img_dtype = torch.float16 if device.type == "cuda" else torch.float32
        for cam_name in self.runtime_params.camera_names:
            norm_obs_state[cam_name] = torch.from_numpy(self.img_obs_history[cam_name] / 255.0).to(
                                            device=device, dtype=img_dtype, non_blocking=True)

        action_mean = torch.from_numpy(self.norm_stats['action']['mean'])
        action_std = torch.from_numpy(self.norm_stats['action']['std'])

        action_tensor = torch.from_numpy(action_chunk).to(dtype=torch.float32)
        # Add batch dimension: (num_queries, action_dim) -> (1, num_queries, action_dim)
        action_tensor = action_tensor.unsqueeze(0)

        normalized = (action_tensor - action_mean.view(1, 1, -1)) / (action_std.view(1, 1, -1) + self.eps)

        return norm_obs_state
    
    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """
        Denormalize action using stats.

        Args:
            action: Normalized action tensor

        Returns:
            Denormalized action as numpy array
        """
        action_mean = torch.from_numpy(self.norm_stats['action']['mean']).to(action.device)
        action_std = torch.from_numpy(self.norm_stats['action']['std']).to(action.device)

        # Denormalize: action = action * std + mean
        denormalized = action * action_std + action_mean
        return denormalized.cpu().numpy()
