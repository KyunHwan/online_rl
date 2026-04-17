class DataManagerInterface:
    def __init__(self, runtime_params, robot):
        self.data_manager_bridge = None
        if robot == "igris_b":
            from env_actor.auto.inference_algorithms.sequential.data_manager.robots.igris_b.data_manager_bridge import DataManagerBridge
        elif robot == "igris_c":
            from env_actor.auto.inference_algorithms.sequential.data_manager.robots.igris_c.data_manager_bridge import DataManagerBridge
        self.data_manager_bridge = DataManagerBridge(runtime_params=runtime_params)

    def update_state_history(self, obs_data):
        self.data_manager_bridge.update_state_history(obs_data)

    def buffer_action_chunk(self, policy_output, t):
        self.data_manager_bridge.buffer_action_chunk(policy_output, t)

    def get_current_action(self, t):
        return self.data_manager_bridge.get_current_action(t)

    def init_inference_obs_state_buffer(self, init_data):
        self.data_manager_bridge.init_inference_obs_state_buffer(init_data)

    def serve_raw_obs_state(self):
        return self.data_manager_bridge.serve_raw_obs_state()

    def serve_init_action(self):
        return self.data_manager_bridge.serve_init_action()
