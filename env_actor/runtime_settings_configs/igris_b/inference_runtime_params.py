from pathlib import Path
import pickle
import numpy as np

class RuntimeParams:
    def __init__(self, inference_runtime_config):
        self._HZ = inference_runtime_config['HZ']
        self._policy_update_period = inference_runtime_config['policy_update_period']

        self._max_delta = np.deg2rad(inference_runtime_config['max_delta_deg'])
        self._proprio_state_dim = inference_runtime_config['proprio_state_dim']
        self._proprio_history_size = inference_runtime_config['proprio_history_size']

        self._camera_names = inference_runtime_config['camera_names']
        self._num_img_obs = inference_runtime_config['num_img_obs']
        self._img_obs_every = inference_runtime_config['img_obs_every']
        self._mono_img_resize_width = inference_runtime_config['mono_image_resize']['width']
        self._mono_img_resize_height = inference_runtime_config['mono_image_resize']['height']

        self._action_dim = inference_runtime_config['action_dim']
        self._action_chunk_size = inference_runtime_config['action_chunk_size']

        self._norm_stats_file_path = inference_runtime_config['norm_stats_file_path']
    
    @property
    def HZ(self):
        return self._HZ

    @property
    def policy_update_period(self):
        return self._policy_update_period

    @property
    def max_delta(self):
        return self._max_delta

    @property
    def proprio_state_dim(self):
        return self._proprio_state_dim

    @property
    def proprio_history_size(self):
        return self._proprio_history_size

    @property
    def camera_names(self):
        return self._camera_names

    @property
    def mono_img_resize_width(self):
        return self._mono_img_resize_width
    
    @property
    def mono_img_resize_height(self):
        return self._mono_img_resize_height
    
    @property
    def action_dim(self):
        return self._action_dim
    
    @property
    def action_chunk_size(self):
        return self._action_chunk_size
    
    @property
    def num_img_obs(self):
        return self._num_img_obs
    
    @property
    def img_obs_every(self):
        return self._img_obs_every
    
    @property
    def norm_stats(self):
        return self._norm_stats
        
    def read_stats_file(self):
        norm_stats = None
        norm_stats_file_path = Path(self._norm_stats_file_path)
        if norm_stats_file_path.is_file():
            with norm_stats_file_path.open('rb') as file:
                norm_stats = pickle.load(file)
        else:
            print(f"File not found at: {norm_stats_file_path}")
        return norm_stats