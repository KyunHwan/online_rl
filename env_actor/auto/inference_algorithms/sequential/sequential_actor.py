"""
Sequential inference engine for robot manipulation.

Orchestrates the main control loop for sequential inference.
Simple orchestration: Controller -> DataManager -> Policy -> DataManager -> Controller
"""

import time
import torch
import numpy as np
from tensordict import TensorDict
import ray

from env_actor.auto.io_interface.controller_interface import ControllerInterface
from env_actor.auto.data_manager.data_manager_interface import DataManagerInterface
from env_actor.episode_recorder.episode_recorder_interface import EpisodeRecorderInterface
from online_rl.env_actor.policy.utils.loader import build_policy
from online_rl.env_actor.policy.utils.weight_transfer import load_state_dict_cpu_into_module

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
        robot_config,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = build_policy(policy_yaml_path=policy_yaml_path, map_location=self.device)
        self.controller_interface = ControllerInterface(robot_config=robot_config, robot=robot)
        self.data_manager_interface = DataManagerInterface(robot_config=robot_config, robot=robot)
        self.episode_recorder = EpisodeRecorderInterface(robot=robot)

        self.policy_state_manager_handle = policy_state_manager_handle
        self.episode_queue_handle = episode_queue_handle

    def start(self) -> None:
        # 2. Start state readers (cameras and proprioception)
        print("Starting state readers...")
        self.controller_interface.start_state_readers()

        # 4. Initialize timing
        rate_controller = self.controller_interface.recorder_rate_controller()
        DT = self.controller_interface.DT
        
        episode = 0

        while True:
            if episode > 0:
                current_weights_ref = self.policy_state_manager_handle.get_weights.remote()
                current_weights = ray.get(current_weights_ref)
                if current_weights is not None:
                    for model_name, model in self.policy.components.items():
                        sd_cpu = current_weights[model_name]   # <-- critical fix
                        missing, unexpected = load_state_dict_cpu_into_module(model, sd_cpu, strict=True)

                # TODO: Serve the train data buffer
                episodic_data_ref = ray.put(TensorDict.stack(self.episode_recorder.serve_train_data_buffer(),
                                                             dim=0))
                self.episode_queue_handle.put(episodic_data_ref,
                                              block=True)
                
            self.data_manager_interface.init_train_data_buffer()

            print("Initializing robot position...")
            prev_joint = self.controller_interface.init_robot_position()
            self.data_manager_interface.update_prev_joint(prev_joint)

            print("Bootstrapping observation history...")
            initial_state = self.controller_interface.read_state()

            self.data_manager_interface.init_inference_obs_state_buffer(initial_state)

            next_t = time.perf_counter()

            # 5. Main control loop
            for t in range(9000):
                # Rate-limit to maintain target HZ
                rate_controller.sleep()

                # a. Read latest observations (raw from robot)
                obs_data = self.controller_interface.read_state()
                self.episode_recorder.add_obs_state(obs_data)

                if 'proprio' not in obs_data:
                    print(f"Warning: No proprio data at step {t}, skipping...")
                    continue

                # b. Update observation history in data manager
                self.data_manager_interface.update_state_history(obs_data)

                # c. Conditionally run policy
                if (t % self.controller_interface.policy_update_period) == 0 or t == 0:
                    # Get normalized observations from data manager
                    normalized_obs = self.data_manager_interface.serve_normalized_obs_state(self.device)

                    # Generate noise in data manager
                    noise = self.data_manager_interface.generate_noise(self.device)

                    # Add noise to observation dict for policy
                    normalized_obs['noise'] = noise

                    # Run policy forward pass (just neural network)
                    policy_output = self.policy.predict(normalized_obs)

                    # Buffer and denormalize action in data manager
                    self.data_manager_interface.buffer_action_chunk(policy_output, t)

                # d. Get current action from data manager (already denormalized)
                action = self.data_manager_interface.get_current_action(t)
                

                # e. Publish action to robot (includes slew-rate limiting)
                smoothed_joints, fingers = self.controller_interface.publish_action(
                    action,
                    self.data_manager_interface.prev_joint
                )
                self.episode_recorder.add_action(np.concatenate([np.concatenate([smoothed_joints[6:], smoothed_joints[:6]]),
                                                                               fingers]))

                # f. Update previous joint state in data manager
                self.data_manager_interface.update_prev_joint(smoothed_joints)

                # g. Maintain precise loop timing
                next_t += DT
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            print("Sequential inference completed successfully!")

            episode += 1


"""
import ray
import torch

@ray.remote(num_gpus=1)
class SequentialActor:
    def __init__(self, 
                 policy_state_manager_handle, 
                 episode_queue_handle,
                 inference_config_path):
        self.policy_state_manager_handle = policy_state_manager_handle
        self.episode_queue_handle = episode_queue_handle
        self.inference_config_path = inference_config_path
        self.policy = None

    def start(self):
        while True:
            current_weights_ref = self.policy_state_manager_handle.get_weights.remote()
            current_weights = ray.get(current_weights_ref)
            if current_weights is not None:
                new_weights = current_weights # Zero-copy fetch
                print("weights updated: ", new_weights.keys())
                #self.policy.load_state_dict(new_weights)



@ray.remote
class ControllerActor:
    def __init__(self, 
                 episode_queue_handle, 
                 inference_config_path, 
                 shared_memory_names):
        self.episode_queue_handle = episode_queue_handle
        self.inference_config_path = inference_config_path
    
    def start(self,):
        episodic_data = []
        while True:
            if len(episodic_data) != 0:
                # wait until there's room to put the data in the queue
                episodic_data_ref = ray.put(TensorDict.stack(episodic_data, dim=0))
                self.episode_queue_handle.put(episodic_data_ref,
                                              block=True)
            episodic_data = []
            for step in range(900):
                episodic_data.append(TensorDict({
                    'reward': torch.randn(40, 24),
                    'action': torch.ones(40,24),
                    'state': torch.zeros(40, 24)
                }, batch_size=[]))
"""