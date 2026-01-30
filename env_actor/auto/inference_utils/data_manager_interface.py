class DataManagerInterface:
    def __init__(self, robot_config, robot):
        self.data_manager_bridge = None
        if robot == "igris_b":
            import env_actor.auto.inference_utils.igris_b.data_manager_bridge as DataPManagerBridge
        elif robot == "igris_c":
            import env_actor.auto.inference_utils.igris_c.data_manager_bridge as DataPManagerBridge
        self.data_manager_bridge = DataPManagerBridge(robot_config)
    
    def update_prev_joint(self, val):
        self.data_manager_bridge.update_prev_joint(val)

    def update_norm_stats(self):
        self.data_manager_bridge.update_norm_stats()

    def update_state_history(self, obs_data):
        self.data_manager_bridge(obs_data)

    def denormalize_action(self, action) -> dict:
        return self.data_manager_bridge.denormalize_action(action)
    
    def serve_normalized_obs_state(self) -> dict:
        return self.data_manager_bridge.normalize_state()
    
    def init_inference_obs_state_buffer(self, init_data):
        self.data_manager_bridge.init_inference_obs_state_buffer(init_data)

    def init_train_data_buffer(self):
        self.data_manager_bridge.init_train_data_buffer()