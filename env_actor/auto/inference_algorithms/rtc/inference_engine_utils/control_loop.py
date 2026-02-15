def start_control(
        robot,
        inference_runtime_params_config,
        inference_runtime_topics_config,

        shm_specs,
        lock,
        control_iter_cond,
        inference_ready_cond,
        stop_event,
        episode_complete_event,
        num_control_iters,
        inference_ready_flag,

        episode_queue_handle_b,
    ):
    """Synchronous implementation of the main control loop.

    Note: Changed from async to sync since SharedMemoryManager uses
    standard multiprocessing primitives, not asyncio.
    """
    import ray
    from ray import cloudpickle
    if not ray.is_initialized():
        ray.init(address="auto", namespace="online_rl", log_to_driver=False)
    
    episode_queue_handle = cloudpickle.loads(episode_queue_handle_b)

    import time
    import json
    import numpy as np

    from env_actor.episode_recorder.episode_recorder_interface import EpisodeRecorderInterface
    from env_actor.auto.io_interface.controller_interface import ControllerInterface
    from ..data_manager.shm_manager_interface import SharedMemoryInterface
    
    # Load robot-specific RuntimeParams
    if robot == "igris_b":
        from env_actor.runtime_settings_configs.igris_b.inference_runtime_params import RuntimeParams
    # elif robot == "igris_c":
    #     from env_actor.runtime_settings_configs.igris_c.inference_runtime_params import RuntimeParams
    else:
        raise ValueError(f"Unknown robot: {robot}")

    

    if isinstance(inference_runtime_params_config, str):
        with open(inference_runtime_params_config, 'r') as f:
            inference_runtime_params_config = json.load(f)
    runtime_params = RuntimeParams(inference_runtime_params_config)

    if isinstance(inference_runtime_topics_config, str):
        with open(inference_runtime_topics_config, 'r') as f:
            inference_runtime_topics_config = json.load(f)

    """Initialize the controller actor."""
    # Initialize interfaces
    controller_interface = ControllerInterface(
                                runtime_params=runtime_params, 
                                inference_runtime_topics_config=inference_runtime_topics_config, robot=robot
                            )

    # Create SharedMemoryManager from specs (attaches to existing SharedMemory)
    shm_manager = SharedMemoryInterface.attach_from_specs(
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

    episode_recorder = EpisodeRecorderInterface(robot=robot)

    # Episode configuration
    episode_length = 100  # Control steps per episode

    try:
        print("Starting state readers...")
        controller_interface.start_state_readers()

        print("Starting control loop...")
        episode = -1

        while True:
            # Check stop event
            if shm_manager.stop_event_is_set():
                print("Stop event received, exiting control loop")
                break

            print("Waiting for inference actor to be ready...")
            if not shm_manager.wait_for_inference_ready():
                print("Stop event received before inference ready, exiting")
                return
            # Clear episode_complete after inference signals ready (ensures handshake)
            shm_manager.clear_episode_complete()

            # Episode boundary handling
            if episode >= 0:
                print(f"Submitting episode {episode} data...")
                sub_eps = episode_recorder.serve_train_data_buffer(episode)
                for sub_ep in sub_eps:
                    sub_ep_data_ref = ray.put(sub_ep)
                    episode_queue_handle.put(sub_ep_data_ref, block=True)

            # Initialize new episode
            episode_recorder.init_train_data_buffer()
            episode += 1
            print(f"Starting episode {episode}...")

            # Reset robot position
            print("Initializing robot position...")
            prev_joint = controller_interface.init_robot_position()
            time.sleep(0.5)

            # Reset SharedMemoryManager for new episode (direct call, no Ray)
            shm_manager.reset()
            shm_manager.init_action_chunk()
            shm_manager.bootstrap_obs_history(obs_history=controller_interface.read_state())

            # Main control loop for episode
            next_t = time.perf_counter()
            print("Control loop started...")
            for t in range(episode_length):
                print(f"step {t}")
                # Check stop event
                if shm_manager.stop_event_is_set():
                    print("Stop event received during episode, exiting")
                    return

                # a. Read latest observations
                obs_data = controller_interface.read_state()
                if "proprio" not in obs_data:
                    print(f"Warning: No proprio data at step {t}, skipping...")
                    continue

                episode_recorder.add_obs_state(obs_data)

                # e. Update SharedMemory (atomic write + increment, direct call)
                action = shm_manager.atomic_write_obs_and_increment_get_action(obs=obs_data, 
                                                                                    action_chunk_size=runtime_params.action_chunk_size)

                # h. Publish action to robot (includes slew-rate limiting)
                smoothed_joints, fingers = controller_interface.publish_action(action, prev_joint)

                recorded_action = np.concatenate([
                    np.concatenate([smoothed_joints[6:], smoothed_joints[:6]]),
                    fingers,
                ])
                episode_recorder.add_action(recorded_action)

                # j. Update previous joint state
                prev_joint = smoothed_joints

                # k. Maintain precise loop timing
                next_t += controller_interface.DT
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            print(f"Episode {episode} finished!")
            shm_manager.signal_episode_complete()
    finally:
        if 'shm_manager' in locals():
            shm_manager.cleanup()
        controller_interface.shutdown()