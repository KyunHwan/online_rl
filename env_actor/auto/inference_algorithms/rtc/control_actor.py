"""ControllerActor - Robot I/O Ray actor for RTC inference.

This actor handles all robot interaction for the RTC inference algorithm:
- Reading robot state (proprioception and camera images)
- Updating shared observation history via SharedMemoryManager
- Executing actions from the shared action chunk
- Recording episodes for online RL training

Key responsibilities:
- Maintain precise control loop timing at target HZ
- Coordinate state updates via SharedMemoryManager (direct SharedMemory access)
- Record episode data via EpisodeRecorderInterface
- Handle episode boundaries and robot reinitialization
"""
from __future__ import annotations

import time
from multiprocessing.synchronize import Condition as ConditionType
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import RLock as RLockType
from typing import TYPE_CHECKING, Any

import numpy as np
import ray
from tensordict import TensorDict

from env_actor.auto.io_interface.controller_interface import ControllerInterface
from env_actor.episode_recorder.episode_recorder_interface import EpisodeRecorderInterface

from .data_manager.shm_manager_interface import SharedMemoryInterface
from .data_manager.utils.utils import ShmArraySpec

if TYPE_CHECKING:
    from ray.actor import ActorHandle


@ray.remote
class ControllerActor:
    """Robot I/O actor for RTC algorithm.

    Runs the control loop at target frequency, coordinating with:
    - SharedMemoryManager: Direct SharedMemory access for observation/action state
    - EpisodeRecorderInterface: Records episode data for training
    - ControllerInterface: Robot I/O abstraction
    - DataManagerInterface: Observation history management

    The control loop:
    1. Reads robot state via ControllerInterface
    2. Updates observation history via SharedMemoryManager
    3. Reads action from shared action chunk
    4. Publishes action to robot
    5. Records data for training

    Args:
        runtime_params: Runtime parameters for control
        inference_runtime_topics_config: Topics configuration for inference
        robot: Robot identifier ("igris_b" or "igris_c")
        episode_queue_handle: Ray handle to episode queue for training
        shm_specs: Dict of ShmArraySpec for SharedMemory blocks
        lock: Shared RLock for atomic operations
        control_iter_cond: Shared Condition for control iteration waits
        inference_ready_cond: Shared Condition for inference ready waits
        stop_event: Shared Event for shutdown signaling
        num_control_iters: Shared Value for control iteration counter
        inference_ready_flag: Shared Value for inference ready signal
    """

    def __init__(
        self,
        runtime_params,
        inference_runtime_topics_config,
        robot: str,
        episode_queue_handle: ActorHandle,
        shm_specs: dict[str: ShmArraySpec],
        lock: RLockType,
        control_iter_cond: ConditionType,
        inference_ready_cond: ConditionType,
        stop_event: EventType,
        episode_complete_event: EventType,
        num_control_iters: Any,  # multiprocessing.Value
        inference_ready_flag: Any,  # multiprocessing.Value
    ):
        """Initialize the controller actor."""
        # Initialize interfaces
        self.runtime_params = runtime_params
        self.controller_interface = ControllerInterface(
            runtime_params=runtime_params, inference_runtime_topics_config=inference_runtime_topics_config, robot=robot
        )
        self.episode_recorder = EpisodeRecorderInterface(robot=robot)

        # Store episode queue handle for training data submission
        self.episode_queue_handle = episode_queue_handle

        # Create SharedMemoryManager from specs (attaches to existing SharedMemory)
        self.shm_manager = SharedMemoryInterface.attach_from_specs(
            robot=robot,
            shm_specs=shm_specs,
            lock=lock,
            control_iter_cond=control_iter_cond,
            inference_ready_cond=inference_ready_cond,
            stop_event=stop_event,
            episode_complete_event=episode_complete_event,
            num_control_iters=num_control_iters,
            inference_ready_flag=inference_ready_flag,
        )

        # Episode configuration
        self.episode_length = 9000  # Control steps per episode

    def start(self) -> None:
        """Main control loop - runs episodes continuously.

        This method runs the RTC control loop:
        1. Initialize robot and observation history
        2. Wait for inference actor to be ready
        3. Initialize SharedMemory with initial data
        4. Run control loop for each episode
        5. Submit episode data to training queue
        """
        try:
            self._sync_start()
        finally:
            # Cleanup SharedMemory on exit
            self.shm_manager.cleanup()

    def _sync_start(self) -> None:
        """Synchronous implementation of the main control loop.

        Note: Changed from async to sync since SharedMemoryManager uses
        standard multiprocessing primitives, not asyncio.
        """

        print("Starting state readers...")
        self.controller_interface.start_state_readers()

        print("Starting control loop...")
        episode = -1

        while True:
            # Check stop event
            if self.shm_manager.stop_event_is_set():
                print("Stop event received, exiting control loop")
                break

            print("Waiting for inference actor to be ready...")
            if not self.shm_manager.wait_for_inference_ready():
                print("Stop event received before inference ready, exiting")
                return
            # Clear episode_complete after inference signals ready (ensures handshake)
            self.shm_manager.clear_episode_complete()

            # Episode boundary handling
            if episode >= 0:
                print(f"Submitting episode {episode} data...")
                sub_eps = self.episode_recorder.serve_train_data_buffer(episode)
                for sub_ep in sub_eps:
                    sub_ep_data_ref = ray.put(sub_ep)
                    self.episode_queue_handle.put(sub_ep_data_ref, block=True)

            # Initialize new episode
            self.episode_recorder.init_train_data_buffer()
            episode += 1
            print(f"Starting episode {episode}...")

            # Reset robot position
            print("Initializing robot position...")
            prev_joint = self.controller_interface.init_robot_position()
            time.sleep(0.5)

            # Reset SharedMemoryManager for new episode (direct call, no Ray)
            self.shm_manager.reset()
            self.shm_manager.init_action_chunk()
            self.shm_manager.bootstrap_obs_history(obs_history=self.controller_interface.read_state())

            # Main control loop for episode
            next_t = time.perf_counter()

            for t in range(self.episode_length):
                # Check stop event
                if self.shm_manager.stop_event_is_set():
                    print("Stop event received during episode, exiting")
                    return

                # a. Read latest observations
                obs_data = self.controller_interface.read_state()

                if "proprio" not in obs_data:
                    print(f"Warning: No proprio data at step {t}, skipping...")
                    continue

                # b. Record observation
                self.episode_recorder.add_obs_state(obs_data)

                # e. Update SharedMemory (atomic write + increment, direct call)
                action = self.shm_manager.atomic_write_obs_and_increment_get_action(obs=obs_data, 
                                                                                    action_chunk_size=self.runtime_params.action_chunk_size)

                # h. Publish action to robot (includes slew-rate limiting)
                smoothed_joints, fingers = self.controller_interface.publish_action(action, prev_joint)

                # i. Record action (reorder to match training format)
                # Format: [L joints 6] + [R joints 6] + [fingers 12]
                recorded_action = np.concatenate([
                    np.concatenate([smoothed_joints[6:], smoothed_joints[:6]]),
                    fingers,
                ])
                self.episode_recorder.add_action(recorded_action)

                # j. Update previous joint state
                prev_joint = smoothed_joints

                # k. Maintain precise loop timing
                next_t += self.controller_interface.DT
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            print(f"Episode {episode} finished!")
            self.shm_manager.signal_episode_complete()
