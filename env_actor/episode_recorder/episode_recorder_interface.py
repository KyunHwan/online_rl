class EpisodeRecorderInterface:
    def __init__(self, robot):
        self.episode_recorder_bridge = None
        if robot == "igris_b":
            import env_actor.episode_recorder.igris_b.episode_recorder_bridge as EpisodeRecorderBridge
        elif robot == "igris_c":
            import env_actor.episode_recorder.igris_c.episode_recorder_bridge as EpisodeRecorderBridge
        self.episode_recorder_bridge = EpisodeRecorderBridge()

    def serve_train_data_buffer(self):
        return self.episode_recorder_bridge.serve_train_data_buffer()
    
    def add_obs_state(self, obs_data):
        self.episode_recorder_bridge.add_obs_state(obs_data)
    
    def add_action(self, action):
        self.episode_recorder_bridge.add_action(action)
