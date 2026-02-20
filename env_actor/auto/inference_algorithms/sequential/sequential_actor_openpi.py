"""Sequential inference engine for pi05_igris (openpi) policy.

Variant of sequential_actor.py that:
- Creates Pi05IgrisVlaAdapter via vla_'s create_trained_policy (not build_policy)
- Uses get_raw_obs_arrays() for raw observations (no z-score normalization)
- Buffers raw denormalized actions directly (no z-score denormalization step)
"""

import time
import torch
import numpy as np
import ray

@ray.remote(num_gpus=1)
class SequentialActorOpenpi:
    """Sequential inference loop using pi05_igris (openpi) policy.

    Orchestration (same as SequentialActor):
    1. Controller reads raw state
    2. DataManager buffers observation history
    3. Policy runs forward pass on raw observations
    4. Actions buffered directly (already denormalized by vla_)
    5. Controller publishes action

    Args:
        runtime_params: RuntimeParams instance with config dimensions
        inference_runtime_topics_config: ROS/comms topic configuration
        robot: Robot identifier ("igris_b" or "igris_c")
        ckpt_dir: Path to vla_ checkpoint dir (contains model.safetensors + assets/)
        default_prompt: Optional language instruction for the policy
    """

    def __init__(
        self,
        runtime_params,
        inference_runtime_topics_config,
        robot,
        ckpt_dir,
        default_prompt=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from env_actor.robot_io_interface.controller_interface import ControllerInterface
        from .data_manager.data_manager_interface import DataManagerInterface
        from env_actor.policy.policies.pi05_igris.pi05_igris import Pi05IgrisVlaAdapter
        
        # Create pi05_igris via vla_'s factory (NOT build_policy)
        self.policy = Pi05IgrisVlaAdapter(
            ckpt_dir=ckpt_dir,
            device=str(self.device),
            default_prompt=default_prompt,
        )
        print(f"[SequentialActorOpenpi] Policy: action_dim={self.policy.action_dim}, "
              f"state_dim={self.policy.state_dim}, action_horizon={self.policy.action_horizon}")

        self.controller_interface = ControllerInterface(
            runtime_params=runtime_params,
            inference_runtime_topics_config=inference_runtime_topics_config,
            robot=robot,
        )
        self.data_manager_interface = DataManagerInterface(
            runtime_params=runtime_params,
            robot=robot,
        )

    def start(self) -> None:
        # Start state readers (cameras + proprioception)
        print("Starting state readers...")
        self.controller_interface.start_state_readers()

        rate_controller = self.controller_interface.recorder_rate_controller()
        DT = self.controller_interface.DT
        episode = -1
        try:
            while True:
                print("Initializing robot position...")
                prev_joint = self.controller_interface.init_robot_position()
                time.sleep(0.5)

                self.data_manager_interface.update_prev_joint(prev_joint)

                print("Bootstrapping observation history...")
                initial_state = self.controller_interface.read_state()
                self.data_manager_interface.init_inference_obs_state_buffer(initial_state)
                print("Data manager interface is ready...")
                next_t = time.perf_counter()

                # Main control loop
                for t in range(9000):
                    print("reading action")
                    rate_controller.sleep()
                    # a. Read latest observations (raw from robot)
                    obs_data = self.controller_interface.read_state()

                    if "proprio" not in obs_data:
                        print(f"Warning: No proprio data at step {t}, skipping...")
                        continue

                    # b. Update observation history in data manager
                    self.data_manager_interface.update_state_history(obs_data)

                    # c. Conditionally run policy
                    if (t % self.controller_interface.policy_update_period) == 0 or t == 0:
                        # Get RAW observations (not z-score normalized)
                        raw_obs = self.data_manager_interface.get_raw_obs_arrays()

                        # Run openpi policy (handles all normalization internally)
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            raw_actions = self.policy.predict(obs=raw_obs, noise=None)
                        # raw_actions: np.float32 (action_horizon, action_dim) -- denormalized

                        expected_shape = (self.policy.action_horizon, self.policy.action_dim)
                        if raw_actions.shape != expected_shape:
                            print(f"[SequentialActorOpenpi] Warning: action shape {raw_actions.shape} "
                                f"!= expected {expected_shape}")

                        # Buffer raw actions directly (bypass data_manager's denormalization)
                        self.data_manager_interface.openpi_buffer_denormalized_action_chunk(
                            policy_output=raw_actions, current_step=t,
                        )

                    # d. Get current action (same logic as original -- simple offset indexing)
                    action = self.data_manager_interface.get_current_action(t)

                    # e. Publish action to robot (includes slew-rate limiting)
                    smoothed_joints, fingers = self.controller_interface.publish_action(
                        action,
                        self.data_manager_interface.prev_joint,
                    )

                    # f. Update previous joint state
                    self.data_manager_interface.update_prev_joint(smoothed_joints)

                    # g. Maintain precise loop timing
                    next_t += DT
                    sleep_time = next_t - time.perf_counter()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

                print("Episode finished !!")
                episode += 1
        finally:
            self.controller_interface.shutdown()
