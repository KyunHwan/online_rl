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
    def __init__(self, inference_runtime_config):
        self.runtime_params = RuntimeParams(inference_runtime_config=inference_runtime_config)

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

        # Train Data Buffer
        self.all_time_data = []
    
    def _denormalize_action_chunk(self, action: torch.Tensor, device: torch.device) -> np.ndarray:
        """
        Denormalize action using stats.

        Args:
            action: Normalized action tensor
            device: Device for tensor operations

        Returns:
            Denormalized action as numpy array
        """


                data['observation.state'] = (data['observation.state'] - stats_cpu['observation.state']['mean']) / (stats_cpu['observation.state']['std'] + 1e-8)
                data['observation.current'] = (data['observation.current'] - stats_cpu['observation.current']['mean']) / (stats_cpu['observation.current']['std'] + 1e-8)
                data['observation.proprio_state'] = data['observation.state']
                data['observation.state'] = torch.concat([data['observation.state'], data['observation.current']], dim=-1)

        action_mean = torch.from_numpy(self.norm_stats['action']['mean']).to(device)
        action_std = torch.from_numpy(self.norm_stats['action']['std']).to(device)

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
        for key in self.img_obs_history.keys():
             if key != "proprio":
                if self.runtime_params.img_obs_every <= 1 or\
                    (self.image_frame_counter % self.runtime_params.img_obs_every == 0):
                    if self.runtime_params.num_img_obs > 1:
                        self.img_obs_history[key][1:] = self.img_obs_history[key][:-1]
                self.img_obs_history[key][0] = obs_data[key]

        self.image_frame_counter += 1
    
    def serve_normalized_obs_state(self, device: torch.device) -> dict[str, torch.Tensor]:
        """
        Serve normalized observations for policy inference.

        Args:
            device: Device for tensor operations

        Returns:
            Dict with 'proprio' and 'images' tensors
        """
        # Load normalization stats
        state_mean = torch.from_numpy(self.norm_stats['state_mean']).to(device)
        state_std = torch.from_numpy(self.norm_stats['state_std']).to(device)
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

        rh = rh.reshape(1, -1).view(1, num_robot_obs, state_dim)

        # Normalize images
        cam_list = []
        for cam_name in self.runtime_params.camera_names:
            cam_list.append(self.img_obs_history[cam_name])
        all_cam_images = np.stack(cam_list, axis=0)

        img_dtype = torch.float16 if device.type == "cuda" else torch.float32
        cam_images = torch.from_numpy(all_cam_images / 255.0).to(
            device=device, dtype=img_dtype, non_blocking=True
        ).unsqueeze(0)

        return {'proprio': rh, 'images': cam_images}
    
    def generate_noise(self, device: torch.device) -> torch.Tensor:
        """
        Generate noise tensor for policy.

        Args:
            device: Device for tensor

        Returns:
            Noise tensor (1, num_queries, action_dim)
        """
        return torch.randn(1, self.num_queries, self.action_dim).to(device)

    def buffer_action_chunk(self, policy_output: torch.Tensor, current_step: int, device: torch.device):
        """
        Denormalize and buffer action chunk from policy.

        Args:
            policy_output: Normalized actions from policy (1, num_queries, action_dim)
            current_step: Current timestep
            device: Device for tensor operations
        """
        # Denormalize action chunk
        denormalized = self._denormalize_action_chunk(policy_output, device)

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

        # Initialize image history buffers
        self.img_obs_history = {
            cam: np.zeros((self.num_image_obs,
                           3,
                           self.runtime_params.mono_img_resize_height,
                           self.runtime_params.mono_img_resize_width), dtype=np.uint8)
            for cam in self.camera_names
        }

        # Initialize proprio history buffer and bootstrap with initial state
        self.robot_proprio_history = np.zeros((self.num_robot_obs,
                                               self.state_dim), dtype=np.float32)

        # Bootstrap history by repeating initial state
        if 'proprio' in init_data:
            init_proprio = init_data['proprio']
            self.robot_proprio_history[:] = np.repeat(
                init_proprio.reshape(1, -1), self.num_robot_obs, axis=0
            )

    def init_train_data_buffer(self):
        """Initialize episodic data buffers for training."""
        self.all_time_data = []

    def get_train_data_buffer(self):

    def _add_episodic_obs_state(self, obs_data):
        """Add observation state to episodic training buffer."""
        # TODO: Implement episodic data collection for training
        pass

    def _add_episodic_action(self, action):
        """Add action to episodic training buffer."""
        # TODO: Implement episodic data collection for training
        pass

    

    
    
    

    

    

    

    
