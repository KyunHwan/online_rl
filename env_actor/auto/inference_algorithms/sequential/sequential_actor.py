"""
Sequential inference engine for robot manipulation.

Orchestrates the main control loop for sequential inference.
Simple orchestration: Controller -> DataManager -> Policy -> DataManager -> Controller
"""


import ray

import time
import torch
import numpy as np
from tensordict import TensorDict

from env_actor.policy.utils.loader import build_policy
from env_actor.policy.utils.weight_transfer import load_state_dict_cpu_into_module

@ray.remote(num_gpus=1)
class SequentialActor:
    """
    Robot-agnostic sequential inference control loop.

    Simple orchestration:
    1. Controller reads raw state
    2. DataManager processes and normalizes
    3. Policy runs forward pass
    4. DataManager denormalizes action
    5. Controller publishes action

    Responsibilities:
    - Control loop timing
    - Policy update scheduling
    - Orchestration (no processing logic)
    """


    def __init__(
        self,
        inference_runtime_params_config,
        inference_runtime_topics_config,
        robot,
        policy_yaml_path,
        policy_state_manager_handle,
        episode_queue_handle,
    ):
        """
        Initialize sequential inference engine.

        Args:
            robot_config: dict read from .json or .yaml file 
                          for igris_b, it's inference_runtime_settings.json in runtime_settings_configs folder
            robot: str ("igris_b" or "igris_c")
            policy_yaml_path: str file path to policy yaml file.
        """
        from env_actor.robot_io_interface.controller_interface import ControllerInterface
        from .data_manager.data_manager_interface import DataManagerInterface
        from env_actor.nom_stats_manager.data_normalization_interface import DataNormalizationInterface
        from env_actor.episode_recorder.episode_recorder_interface import EpisodeRecorderInterface
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot = robot
        # Load robot-specific RuntimeParams
        if self.robot == "igris_b":
            from env_actor.runtime_settings_configs.robots.igris_b.inference_runtime_params import RuntimeParams
        elif self.robot == "igris_c":
            from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams
        else:
            raise ValueError(f"Unknown robot: {self.robot}")
        self.runtime_params = RuntimeParams(inference_runtime_params_config)
        self.policy = build_policy(policy_yaml_path=policy_yaml_path, map_location="cpu").to(self.device)
        self.policy.eval()
        self.controller_interface = ControllerInterface(runtime_params=self.runtime_params, 
                                                        inference_runtime_topics_config=inference_runtime_topics_config,
                                                        robot=robot)
        self.data_manager_interface = DataManagerInterface(runtime_params=self.runtime_params, robot=robot)
        self.data_normalization_interface = DataNormalizationInterface(robot=robot, data_stats=self.runtime_params.read_stats_file())
        self.episode_recorder = EpisodeRecorderInterface(robot=robot)

        self.policy_state_manager_handle = policy_state_manager_handle
        self.episode_queue_handle = episode_queue_handle

    def start(self) -> None:
        # 2. Start state readers (cameras and proprioception)
        print("Starting state readers...")
        self.controller_interface.start_state_readers()

        # 4. Initialize timing
        # rate controller should probably be replaced with using DT since we don't know if it will be used in Igris-C control.
        # rate_controller = self.controller_interface.recorder_rate_controller()
        DT = self.controller_interface.DT
        
        episode = -1

        # Warm up CUDA (once, outside all loops)
        print("Warming up CUDA kernels...")
        with torch.no_grad():
            try:
                self.policy.warmup()
            except Exception as e:
                print(f"Warmup encountered error (may be expected for minimal inputs): {e}")

        while True:
            if episode >= 0:
                # Signal ready for new episode
                current_weights = ray.get(self.policy_state_manager_handle.get_state.remote())
                if current_weights is not None:
                    for model_name in current_weights.keys():
                        if model_name in self.policy.components.keys():
                            missing, unexpected = load_state_dict_cpu_into_module(self.policy.components[model_name], 
                                                                                current_weights[model_name], 
                                                                                strict=True)
                            print(f"{model_name} weights updated")
                    print("Policy weights updated successfully")

                sub_eps = self.episode_recorder.serve_train_data_buffer(episode)
                for sub_ep in sub_eps:
                    sub_ep_data_ref = ray.put(sub_ep)
                    self.episode_queue_handle.put(sub_ep_data_ref, block=True)

            self.episode_recorder.init_train_data_buffer()

            print("Initializing robot position...")
            prev_joint = self.controller_interface.init_robot_position()
            time.sleep(0.5)

            print("Bootstrapping observation history...")
            initial_state = self.controller_interface.read_state()

            self.data_manager_interface.init_inference_obs_state_buffer(initial_state)

            next_t = time.perf_counter()

            # 5. Main control loop
            for t in range(9000):

                # a. Read latest observations (raw from robot)
                obs_data = self.controller_interface.read_state()
                self.episode_recorder.add_obs_state(obs_data)

                if 'proprio' not in obs_data:
                    print(f"Warning: No proprio data at step {t}, skipping...")
                    continue

                # b. Update observation history in data manager
                self.data_manager_interface.update_state_history(obs_data)

                # c. Conditionally run policy
                if (t % self.controller_interface.policy_update_period) == 0:
                    # Get observations from data manager
                    obs = self.data_manager_interface.serve_raw_obs_state()

                    # Run policy forward pass (just neural network)
                    # Normalize inside the policy
                    policy_output = self.policy.predict(obs, self.data_normalization_interface)

                    denormalized_policy_output = self.data_normalization_interface.denormalize_action(policy_output)

                    # Buffer denormalized action in data manager
                    self.data_manager_interface.buffer_action_chunk(denormalized_policy_output, t)

                # d. Get current action from data manager (already denormalized)
                action = self.data_manager_interface.get_current_action(t)

                # e. Publish action to robot (includes slew-rate limiting)
                smoothed_joints, fingers = self.controller_interface.publish_action(
                                                                        action,
                                                                        prev_joint
                                                                    )
                self.episode_recorder.add_action(
                    np.concatenate(
                        [np.concatenate(
                            [smoothed_joints[6:], smoothed_joints[:6]]),
                             fingers]
                    )
                )

                # f. Update previous joint state in data manager
                prev_joint = smoothed_joints

                # g. Maintain precise loop timing
                next_t += DT
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            print("Episode finished !!")

            episode += 1
