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
    Stateful data manager on the controller side
    """
    def __init__(self, inference_runtime_config):
        self.runtime_params = RuntimeParams(inference_runtime_config=inference_runtime_config)
        
        # Inference Data Buffer
        self.norm_stats = self.runtime_params.read_stats_file()
        self.img_obs_history = None
        self.robot_proprio_history = None
        self.image_frame_counter = 0
        self.prev_joint = None

        # Train Data Buffer
        self.all_time_action = None
        self.all_time_img = None
        self.all_time_proprio_state = None
        
    def update_norm_stats(self):
        # TODO

    def update_prev_joint(self, val):
        # prev joint needs to be initialized via init_robot_position method of Controller Bridge/Interface method
        self.prev_joint = val

    def add_episodic_obs_state(self, obs_data):

    def add_episodic_action(self, action):

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

    def denormalize_action(self, action):
        denormalized_action = {'action': None}
        return denormalized_action
    
    def serve_normalized_obs_state(self):
        # proprio

        rh = torch.from_numpy(robot_obs_history).to(device=device, dtype=torch.float32, non_blocking=True)
        if sm.numel() == state_dim and ss.numel() == state_dim:
            rh = (rh - sm.view(1, -1)) / (ss.view(1, -1) + eps)
        else:
            # In case stats are flattened differently, repeat to match shape
            sm_rep = sm if sm.numel() == num_robot_obs * state_dim else sm.repeat((num_robot_obs * state_dim + sm.numel() - 1) // sm.numel())[:num_robot_obs * state_dim]
            ss_rep = ss if ss.numel() == num_robot_obs * state_dim else ss.repeat((num_robot_obs * state_dim + ss.numel() - 1) // ss.numel())[:num_robot_obs * state_dim]
            rh = (rh - sm_rep.view(num_robot_obs, state_dim)) / (ss_rep.view(num_robot_obs, state_dim) + eps)
        rh = rh.reshape(1, -1).view(1, policy.num_robot_observations, policy.state_dim)

        torch.from_numpy(self.robot_proprio_history)
        normalized_data = {
            "proprio": torch.from_numpy(self.robot_proprio_history)
        }

        # images
        img_dtype = torch.float16 if device.type == "cuda" else torch.float32
        cam_images = torch.from_numpy(all_cam_images / 255.0).to(
            device=device, dtype=img_dtype, non_blocking=True
        ).unsqueeze(0)
        

        return normalized_data
    
    def init_inference_obs_state_buffer(self, init_data):
        # Inference Data Buffer
        # Should be called AFTER init_robot_position has been set from Controller Bridge/Interface
        # TODO: Need to fill the images and proprio state history with initial data at the start.
        self.image_frame_counter = 0
        self.img_obs_history = {
            cam: np.zeros((self.runtime_params.num_img_obs, 
                           3, 
                           self.runtime_params.mono_img_resize_height, 
                           self.runtime_params.mono_img_resize_width), dtype=np.uint8)
            for cam in self.runtime_params.camera_names
        }

        self.robot_proprio_history = np.zeros((self.runtime_params.proprio_history_size, 
                                           self.runtime_params.proprio_state_dim), dtype=np.float32)

    def init_train_data_buffer(self):
        self.all_time_action = []
        self.all_time_img = []
        self.all_time_proprio_state = []