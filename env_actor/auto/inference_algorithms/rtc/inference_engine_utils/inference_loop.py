import numpy as np

def start_inference(
        robot,
        ckpt_dir,
        default_prompt,
        policy_yaml_path,
        min_num_actions_executed,
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
    ) -> None:
    import ray
    if not ray.is_initialized():
        ray.init(address="auto", namespace="online_rl", log_to_driver=True)

    import torch
    import json

    from ..data_manager.data_normalization_interface import DataNormalizationInterface
    from ..data_manager.shm_manager_interface import SharedMemoryInterface
    from env_actor.policy.utils.weight_transfer import load_state_dict_cpu_into_module
    from env_actor.policy.utils.loader import build_policy
    
    try:
        # Load robot-specific RuntimeParams
        if robot == "igris_b":
            from env_actor.runtime_settings_configs.robots.igris_b.inference_runtime_params import RuntimeParams
        elif robot == "igris_c":
            from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams
        else:
            raise ValueError(f"Unknown robot: {robot}")

        if isinstance(inference_runtime_params_config, str):
            with open(inference_runtime_params_config, 'r') as f:
                inference_runtime_params_config = json.load(f)
        runtime_params = RuntimeParams(inference_runtime_params_config)


        if isinstance(inference_runtime_topics_config, str):
            with open(inference_runtime_topics_config, 'r') as f:
                inference_runtime_topics_config = json.load(f)

        """Main inference loop implementation.

        Note: Changed from async to sync since SharedMemoryManager uses
        standard multiprocessing primitives, not asyncio.

        Structure:
        - Warmup is called once (outside all loops)
        - Outer loop handles per-episode transitions
        - Inner loop handles inference iterations within an episode
        """

        """Initialize the inference actor."""
        # Set up device and CUDA optimizations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # Build policy using env_actor loader
        policy = build_policy(
            policy_yaml_path=policy_yaml_path,
            map_location=device,
        )
        policy.eval()
        
        # Warm up CUDA (once, outside all loops)
        print("Warming up CUDA kernels...")
        with torch.no_grad():
            try:
                policy.warmup()
            except Exception as e:
                print(f"Warmup encountered error (may be expected for minimal inputs): {e}")

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

        policy_state_manager_handle = ray.get_actor("policy_state_manager")

        data_normalization_bridge = DataNormalizationInterface(robot=robot, data_stats=runtime_params.read_stats_file())

        while True:  # Outer loop - per episode
            # Signal ready for new episode
            current_weights = ray.get(policy_state_manager_handle.get_state.remote())
            if current_weights is not None:
                for model_name in current_weights.keys():
                    if model_name in policy.components.keys():
                        missing, unexpected = load_state_dict_cpu_into_module(policy.components[model_name], 
                                                                            current_weights[model_name], 
                                                                            strict=True)
                        print(f"{model_name} weights updated")
                print("Policy weights updated successfully")

            print("Signaling inference ready...")
            shm_manager.set_inference_ready()

            while True:  # Inner loop - inference iterations within episode
                # Wait for minimum actions executed (blocks until threshold, episode_complete, or stop)
                result = shm_manager.wait_for_min_actions(min_num_actions_executed)

                if result == 'stop':
                    print("Stop event received, exiting inference loop")
                    return  # Exit completely

                if result == 'episode_complete':
                    print("Episode complete, waiting for next episode")
                    break  # Exit inner loop, continue to outer loop

                # result == 'min_actions' - proceed with inference
                shm_manager.set_inference_not_ready()

                # Atomically read state from SharedMemory
                # should be torch tensor
                input_data = shm_manager.atomic_read_for_inference()

                with torch.inference_mode() and torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # TODO: Normalize observations and prev_action_chunk
                    # TODO: Denormalize action output
                    #input_data['normalized_proprio'] = data_normalization_bridge.normalize_state()
                    next_actions = policy.guided_inference(input_data)

                shm_manager.write_action_chunk_n_update_iter_val(
                    next_actions, input_data['num_control_iters']
                )
    finally:
        if 'shm_manager' in locals():
            shm_manager.cleanup()