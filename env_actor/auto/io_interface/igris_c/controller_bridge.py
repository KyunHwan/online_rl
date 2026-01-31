import utils

class ControllerBridge:
    def __init__(self, robot_config):
        self.robot_config = robot_config

    def read_state(self) -> dict:
        state_dict = {}
        return state_dict
    
    def write_action(self):
        return None