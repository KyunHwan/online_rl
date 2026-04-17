from typing import Any
import numpy as np
import torch
from env_actor.runtime_settings_configs.robots.igris_b.inference_runtime_params import RuntimeParams
from env_actor.runtime_settings_configs.robots.igris_b.init_params import (
    INIT_JOINT_LIST,
    INIT_HAND_LIST,
    INIT_JOINT,
    IGRIS_B_STATE_KEYS
)

class DataManagerBridge:
    """
    Stateful data manager - handles ALL data processing.

    Responsibilities:
    - Manage observation history buffers
    - Buffer action chunks and select current action
    - Provide dimension info to other components
    """
    def __init__(self, runtime_params):
        self.runtime_params = runtime_params

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

        # Action buffering for sequential inference
        self.last_action_chunk = None  # Last policy output (denormalized)
        self.last_policy_step = -1

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

    def buffer_action_chunk(self, policy_output: torch.Tensor, current_step: int):
        """
        Denormalize and buffer action chunk from policy.

        Args:
            policy_output: Denormalized actions from policy (1, num_queries, action_dim)
            current_step: Current timestep
            device: Device for tensor operations
        """

        # Store as numpy array: (num_queries, action_dim)
        self.last_action_chunk = policy_output.squeeze(0).cpu().numpy() if policy_output.ndim == 3 else policy_output.cpu().numpy()
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

    def serve_raw_obs_state(self) -> dict:
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



    

    

    
