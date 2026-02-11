class EpisodeRecorderInterface:
    def __init__(self, robot):
        self.episode_recorder_bridge = None
        if robot == "igris_b":
            from env_actor.episode_recorder.igris_b.episode_recorder_bridge import EpisodeRecorderBridge
        elif robot == "igris_c":
            from env_actor.episode_recorder.igris_c.episode_recorder_bridge import EpisodeRecorderBridge
        self.episode_recorder_bridge = EpisodeRecorderBridge()

    def serve_train_data_buffer(self, episode_id):
        return self.episode_recorder_bridge.serve_train_data_buffer(episode_id)
    
    def add_obs_state(self, obs_data):
        self.episode_recorder_bridge.add_obs_state(obs_data)
    
    def add_action(self, action, **kwargs):
        self.episode_recorder_bridge.add_action(action, **kwargs)

    def init_train_data_buffer(self):
        self.episode_recorder_bridge.init_train_data_buffer()
