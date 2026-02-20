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

from env_actor.human_in_the_loop.io_interface.controller_interface import ControllerInterface
from .data_manager.data_manager_interface import DataManagerInterface
from env_actor.episode_recorder.episode_recorder_interface import EpisodeRecorderInterface
from env_actor.policy.utils.loader import build_policy
from env_actor.policy.utils.weight_transfer import load_state_dict_cpu_into_module

# Teleoperation
from env_actor.human_in_the_loop.action_mux.teleop_provider import IgrisBTeleopProvider
from env_actor.human_in_the_loop.action_mux.intervention_switch import PedalInterventionSwitch
from env_actor.human_in_the_loop.action_mux.action_mux import ActionMux

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
        runtime_params,
        inference_runtime_topics_config,
        robot,
        policy_yaml_path,
        policy_state_manager_handle,
        episode_queue_handle,
        operator_name="default",
    ):
        """
        Initialize sequential inference engine.

        Args:
            runtime_params: RuntimeParams object with HZ, action_dim, etc.
            inference_runtime_topics_config: dict with ROS topic config
            robot: str ("igris_b" or "igris_c")
            policy_yaml_path: str file path to policy yaml file.
            policy_state_manager_handle: Ray actor handle for weight updates
            episode_queue_handle: Ray queue handle for episode data
            operator_name: str operator name for Manus glove calibration
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = build_policy(policy_yaml_path=policy_yaml_path, map_location="cpu").to(self.device)
        self.controller_interface = ControllerInterface(runtime_params=runtime_params,
                                                        inference_runtime_topics_config=inference_runtime_topics_config,
                                                        robot=robot)
        self.data_manager_interface = DataManagerInterface(runtime_params=runtime_params, robot=robot)
        self.episode_recorder = EpisodeRecorderInterface(robot=robot)

        self.policy_state_manager_handle = policy_state_manager_handle
        self.episode_queue_handle = episode_queue_handle

        # ActionMux: instant switching between policy and teleop
        robot_id = inference_runtime_topics_config["robot_id"]
        teleop_provider = IgrisBTeleopProvider(
            self.controller_interface.ros_executor,
            operator_name=operator_name,
        )
        intervention_switch = PedalInterventionSwitch(
            self.controller_interface.ros_node,
            robot_id,
        )
        self.action_mux = ActionMux(teleop_provider, intervention_switch)

    def start(self) -> None:
        # 2. Start state readers (cameras and proprioception)
        print("Starting state readers...")
        self.controller_interface.start_state_readers()

        # 4. Initialize timing
        # rate controller should probably be replaced with using DT since we don't know if it will be used in Igris-C control.
        # rate_controller = self.controller_interface.recorder_rate_controller()
        DT = self.controller_interface.DT
        
        episode = -1

        while True:
            if episode >= 0:
                current_weights_ref = self.policy_state_manager_handle.get_state.remote()
                current_weights = ray.get(current_weights_ref)
                if current_weights is not None:
                    for model_name, model in self.policy.components.items():
                        sd_cpu = current_weights[model_name]   # <-- critical fix
                        missing, unexpected = load_state_dict_cpu_into_module(model, sd_cpu, strict=True)

                sub_eps = self.episode_recorder.serve_train_data_buffer(episode)
                for sub_ep in sub_eps:
                    sub_ep_data_ref = ray.put(sub_ep)
                    self.episode_queue_handle.put(sub_ep_data_ref, block=True)
                
            self.episode_recorder.init_train_data_buffer()

            print("Initializing robot position...")
            prev_joint = self.controller_interface.init_robot_position()
            time.sleep(0.5)
            
            self.data_manager_interface.update_prev_joint(prev_joint)

            print("Bootstrapping observation history...")
            initial_state = self.controller_interface.read_state()

            self.data_manager_interface.init_inference_obs_state_buffer(initial_state)

            next_t = time.perf_counter()

            # 5. Main control loop
            for t in range(9000):
                # Rate-limit to maintain target HZ
                # rate_controller.sleep()

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

                # d. Get policy action from data manager (always runs, even in teleop)
                policy_action = self.data_manager_interface.get_current_action(t)

                # e. Select action via ActionMux (never blocks)
                action, control_mode = self.action_mux.select(policy_action)

                # f. Publish selected action to robot (includes slew-rate limiting)
                smoothed_joints, fingers = self.controller_interface.publish_action(
                    action, self.data_manager_interface.prev_joint
                )

                # g. Record actual action sent to robot + metadata
                actual_action = np.concatenate([
                    np.concatenate([smoothed_joints[6:], smoothed_joints[:6]]),
                    fingers,
                ])

                self.episode_recorder.add_action(
                    actual_action,
                    control_mode=int(control_mode),
                )

                # f. Update previous joint state in data manager
                self.data_manager_interface.update_prev_joint(smoothed_joints)

                # g. Maintain precise loop timing
                next_t += DT
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            print("Episode finished !!")
            self.action_mux.set_control_mode_to_policy()
            episode += 1