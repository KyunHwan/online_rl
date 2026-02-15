class ControllerInterface:
    def __init__(self, 
                 runtime_params, 
                 inference_runtime_topics_config,
                 robot):
        self.controller_bridge = None
        if robot == "igris_b":
            from env_actor.auto.io_interface.igris_b.controller_bridge import ControllerBridge
        elif robot == "igris_c":
            from env_actor.auto.io_interface.igris_c.controller_bridge import ControllerBridge
        self.controller_bridge = ControllerBridge(runtime_params=runtime_params, 
                                                  inference_runtime_topics_config=inference_runtime_topics_config,)

    @property
    def DT(self):
        return self.controller_bridge.DT
    
    @property
    def policy_update_period(self):
        return self.controller_bridge.policy_update_period

    def recorder_rate_controller(self):
        return self.controller_bridge.recorder_rate_controller()

    def start_state_readers(self):
        self.controller_bridge.start_state_readers()

    def init_robot_position(self):
        return self.controller_bridge.init_robot_position()

    def read_state(self,) -> dict:
        return self.controller_bridge.read_state()

    def publish_action(self, action, prev_joint):
        return self.controller_bridge.publish_action(action, prev_joint)
    
    def shutdown(self):
        self.controller_bridge.shutdown()
    
