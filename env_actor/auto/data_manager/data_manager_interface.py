class DataManagerInterface:
    def __init__(self, robot_config, robot):
        self.data_manager_bridge = None
        if robot == "igris_b":
            import env_actor.auto.data_manager.igris_b.data_manager_bridge as DataManagerBridge
        elif robot == "igris_c":
            import env_actor.auto.data_manager.igris_c.data_manager_bridge as DataManagerBridge
        self.data_manager_bridge = DataManagerBridge(robot_config)

    @property
    def prev_joint(self):
        return self.data_manager_bridge.prev_joint

    def update_prev_joint(self, val):
        self.data_manager_bridge.update_prev_joint(val)

    def update_norm_stats(self):
        self.data_manager_bridge.update_norm_stats()

    def update_state_history(self, obs_data):
        self.data_manager_bridge.update_state_history(obs_data)

    def serve_normalized_obs_state(self, device) -> dict:
        return self.data_manager_bridge.serve_normalized_obs_state(device)

    def generate_noise(self, device):
        return self.data_manager_bridge.generate_noise(device)

    def buffer_action_chunk(self, policy_output, t, device):
        self.data_manager_bridge.buffer_action_chunk(policy_output, t, device)

    def get_current_action(self, t):
        return self.data_manager_bridge.get_current_action(t)

    def init_inference_obs_state_buffer(self, init_data):
        self.data_manager_bridge.init_inference_obs_state_buffer(init_data)

    def init_train_data_buffer(self):
        self.data_manager_bridge.init_train_data_buffer()
    
    def serve_train_data_buffer(self):
        self.data_manager_bridge.serve_train_data_buffer()