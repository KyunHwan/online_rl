from typing import Any
import numpy as np
import torch
from env_actor.runtime_settings_configs.igris_b.inference_runtime_params import RuntimeParams
from env_actor.runtime_settings_configs.igris_b.init_params import (
    INIT_JOINT_LIST,
    INIT_HAND_LIST,
    INIT_JOINT,
    IGRIS_B_STATE_KEYS
)

class DataManagerBridge:
    """
    Stateful data manager - handles ALL data processing.

    Responsibilities:
    - Load and manage normalization stats
    - Manage observation history buffers
    - Normalize observations for policy
    - Denormalize actions from policy
    - Buffer action chunks and select current action
    - Provide dimension info to other components
    """
    def __init__(self, runtime_params):
        self.runtime_params = runtime_params

        # Load normalization stats from file
        self.norm_stats = self.runtime_params.read_stats_file()

        # Extract dimensions from stats and config
        # Dimensions should be loaded from stats file if available, else from config
        if 'dimensions' in self.norm_stats:
            dims = self.norm_stats['dimensions']
            self.num_robot_obs = dims.get('num_robot_obs', self.runtime_params.proprio_history_size)
            self.num_image_obs = dims.get('num_image_obs', self.runtime_params.num_img_obs)
            self.num_queries = dims.get('num_queries', self.runtime_params.action_chunk_size)
            # state_dim and action_dim from stats
            self.state_dim = dims.get('state_dim', self.runtime_params.proprio_state_dim)
            self.action_dim = dims.get('action_dim', self.runtime_params.action_dim)
        else:
            # Fall back to runtime params
            self.num_robot_obs = self.runtime_params.proprio_history_size
            self.num_image_obs = self.runtime_params.num_img_obs
            self.num_queries = self.runtime_params.action_chunk_size
            self.state_dim = self.runtime_params.proprio_state_dim
            self.action_dim = self.runtime_params.action_dim

        self.camera_names = self.runtime_params.camera_names
        self.eps = 1e-8

        # Inference Data Buffers
        self.img_obs_history = None
        self.robot_proprio_history = None
        self.image_frame_counter = 0
        self.prev_joint = None

        # Action buffering for sequential inference
        self.last_action_chunk = None  # Last policy output (denormalized)
        self.last_policy_step = -1

    @property
    def prev_joint(self):
        return self.prev_joint
    
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

    def update_prev_joint(self, val):
        """Update previous joint state for slew-rate limiting."""
        # prev joint needs to be initialized via init_robot_position method of Controller Bridge/Interface method
        self.prev_joint = val

    def update_norm_stats(self):
        """Update normalization stats (placeholder for future dynamic stats updates)."""
        # TODO: Implement dynamic stats updating if needed
        pass

    def update_state_history(self, obs_data):
        """
        data is directly sent from read_state of Controller Bridge/Interface
        """

        # proprio
        if self.runtime_params.proprio_history_size > 1:
            self.robot_proprio_history[1:] = self.robot_proprio_history[:-1]
        self.robot_proprio_history[0] = obs_data['proprio']

        # images
        for cam_name in self.camera_names:
            if self.runtime_params.img_obs_every <= 1 or\
                (self.image_frame_counter % self.runtime_params.img_obs_every == 0):
                if self.runtime_params.num_img_obs > 1:
                    self.img_obs_history[cam_name][1:] = self.img_obs_history[cam_name][:-1]
            self.img_obs_history[cam_name][0] = obs_data[cam_name]

        self.image_frame_counter += 1
    
    def serve_normalized_obs_state(self, device: torch.device) -> dict[str, torch.Tensor]:
        """
        Serve normalized observations for policy inference.

        Args:
            device: Device for tensor operations

        Returns:
            Dict with 'proprio' and 'images' tensors
        """

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

        return norm_obs_state
    
    def generate_noise(self, device: torch.device) -> torch.Tensor:
        """
        Generate noise tensor for policy.

        Args:
            device: Device for tensor

        Returns:
            Noise tensor (1, num_queries, action_dim)
        """
        return torch.randn(1, self.num_queries, self.action_dim).to(device)

    def buffer_action_chunk(self, policy_output: torch.Tensor, current_step: int):
        """
        Denormalize and buffer action chunk from policy.

        Args:
            policy_output: Normalized actions from policy (1, num_queries, action_dim)
            current_step: Current timestep
            device: Device for tensor operations
        """
        # Denormalize action chunk
        denormalized = self.denormalize_action(policy_output)

        # Store as numpy array: (num_queries, action_dim)
        self.last_action_chunk = denormalized.squeeze(0) if denormalized.ndim == 3 else denormalized
        self.last_policy_step = current_step

    def get_current_action(self, current_step: int) -> np.ndarray:
        """
        Get action for current timestep via simple indexing.

        Args:
            current_step: Current timestep in control loop

        Returns:
            Action array (action_dim,)
        """
        if self.last_action_chunk is None:
            raise ValueError("No action chunk available. Call buffer_action_chunk first.")

        # Simple indexing strategy: use offset from last policy update
        offset = current_step - self.last_policy_step
        idx = int(np.clip(offset, 0, self.last_action_chunk.shape[0] - 1))
        action = self.last_action_chunk[idx]
        return action
    
    def init_inference_obs_state_buffer(self, init_data):
        """
        Initialize observation history buffers with initial state.

        Args:
            init_data: Initial observation dict from controller
                Should contain 'proprio' and camera image keys
        """
        self.image_frame_counter = 0
        self.last_policy_step = -1

        # Initialize image history buffers
        self.img_obs_history = {
            cam: np.repeat(
                    init_data[cam][np.newaxis, ...], # Add batch dim: (1, 3, H, W)
                    self.num_image_obs,              # Repeat count
                    axis=0                           # Axis to repeat along
                )
            for cam in self.camera_names
        }

        # Initialize proprio history buffer and bootstrap with initial state
        self.robot_proprio_history = np.repeat(
                                        init_data['proprio'][np.newaxis, ...], # Add batch dim: (1, proprio_dim)
                                        self.num_robot_obs,              # Repeat count
                                        axis=0                           # Axis to repeat along
                                    )

    def normalize_action_chunk(self, action_chunk: np.ndarray) -> torch.Tensor:
        """Normalize action chunk for RTC guided inference.

        Args:
            action_chunk: Action chunk array (num_queries, action_dim)
            device: Torch device for output tensor

        Returns:
            Normalized action chunk as torch tensor (1, num_queries, action_dim)
        """
        action_mean = torch.from_numpy(self.norm_stats['action']['mean'])
        action_std = torch.from_numpy(self.norm_stats['action']['std'])

        action_tensor = torch.from_numpy(action_chunk).to(dtype=torch.float32)
        # Add batch dimension: (num_queries, action_dim) -> (1, num_queries, action_dim)
        action_tensor = action_tensor.unsqueeze(0)

        normalized = (action_tensor - action_mean.view(1, 1, -1)) / (action_std.view(1, 1, -1) + self.eps)
        return normalized

    def get_raw_obs_arrays(self) -> dict:
        """Return raw observation arrays for RTC state sharing.

        Returns:
            Dict with:
            - 'robot_obs_history': (num_robot_obs, state_dim) proprio history
            - 'cam_images': (num_cams, num_img_obs, 3, H, W) camera images
        """
        raw_obs = {
            'proprio': self.robot_proprio_history.copy()
        }
        for cam in self.camera_names:
            raw_obs[cam] = self.img_obs_history[cam]

        return raw_obs
    
    def serve_init_action(self,):
        """ Serve init action for RTC Guided Inference """
        init_vec = np.asarray(
            INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6] + INIT_HAND_LIST[:6] + INIT_HAND_LIST[6:],
            dtype=np.float32,
        )
        # Convert joints to radians, scale fingers
        init_vec[:12] *= np.pi / 180.0
        init_vec[12:] *= 0.03

        # Repeat across all rows
        return np.tile(init_vec, (self.runtime_params.action_chunk_size, 1))



    

    

    
