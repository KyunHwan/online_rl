
import ray
import numpy as np
import json

from ctypes import c_bool

#from multiprocessing import Process, resource_tracker, Condition, Event, RLock, Value

@ray.remote(num_gpus=1, num_cpus=4)
class RTCActorOpenpi:
    def __init__(self,
                 robot,
                 policy_yaml_path,
                 inference_runtime_params_config,
                 inference_runtime_topics_config,
                 min_num_actions_executed,

                 episode_queue_handle,

                 ckpt_dir,
                 default_prompt=None,
                 ):
        # Standard
        self.robot = robot
        self.policy_yaml_path = policy_yaml_path
        self.inference_runtime_params_config = inference_runtime_params_config
        self.inference_runtime_topics_config = inference_runtime_topics_config
        self.min_num_actions_executed = min_num_actions_executed

        # Ray
        self.episode_queue_handle = episode_queue_handle

        # Openpi
        self.ckpt_dir = ckpt_dir
        self.default_prompt = default_prompt
    
    def start(self):
        from ray import cloudpickle
        import multiprocessing as mp
        from multiprocessing import resource_tracker
        from env_actor.auto.inference_algorithms.rtc.data_manager.utils.utils import create_shared_ndarray
        from .inference_engine_utils.control_loop import start_control
        from .inference_engine_utils.inference_loop_openpi import start_inference

        # Load robot-specific RuntimeParams
        if self.robot == "igris_b":
            from env_actor.runtime_settings_configs.igris_b.inference_runtime_params import RuntimeParams
        # elif robot == "igris_c":
        #     from env_actor.runtime_settings_configs.igris_c.inference_runtime_params import RuntimeParams
        else:
            raise ValueError(f"Unknown robot: {self.robot}")

        robot_obs_history_dtype = np.float32
        cam_images_dtype = np.uint8
        action_chunk_dtype = np.float32

        ctx = mp.get_context("spawn")

        if isinstance(self.inference_runtime_params_config, str):
            with open(self.inference_runtime_params_config, 'r') as f:
                self.inference_runtime_params_config = json.load(f)
        runtime_params = RuntimeParams(self.inference_runtime_params_config)

        # Create SharedMemory blocks in parent process
        rob_shm, _, rob_spec = create_shared_ndarray(
            (runtime_params.proprio_history_size, runtime_params.proprio_state_dim), robot_obs_history_dtype
        )
        head_cam_shm, _, head_cam_spec = create_shared_ndarray(
            (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        )
        left_cam_shm, _, left_cam_spec = create_shared_ndarray(
            (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        )
        right_cam_shm, _, right_cam_spec = create_shared_ndarray(
            (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        )
        act_shm, _, act_spec = create_shared_ndarray(
            (runtime_params.action_chunk_size, runtime_params.action_dim), action_chunk_dtype
        )

        shm_specs = {
            "proprio": rob_spec,
            "head": head_cam_spec,
            "left": left_cam_spec,
            "right": right_cam_spec,
            "action": act_spec
        }

        # Create synchronization primitives
        lock = ctx.RLock()
        control_iter_cond = ctx.Condition(lock)      # For num_control_iters waits
        inference_ready_cond = ctx.Condition(lock)   # For inference_ready waits
        stop_event = ctx.Event()
        episode_complete_event = ctx.Event()         # For episode completion signaling
        num_control_iters = ctx.Value('i', 0, lock=False)
        inference_ready_flag = ctx.Value(c_bool, False, lock=False)
        
        # serialize ray handles to bytes
        episode_queue_handle_b  = cloudpickle.dumps(self.episode_queue_handle)

        # Pass ONLY specs to children; they will attach by name
        inference_runner = ctx.Process(
            target=start_inference,
            args=(
                self.robot,
                self.ckpt_dir,
                self.default_prompt,
                self.policy_yaml_path,
                self.min_num_actions_executed,
                self.inference_runtime_params_config,
                self.inference_runtime_topics_config,
                shm_specs,
                lock,
                control_iter_cond,
                inference_ready_cond,
                stop_event,
                episode_complete_event,
                num_control_iters,
                inference_ready_flag,
            ),
            daemon=False
        )
        controller = ctx.Process(
            target=start_control,
            args=(
                self.robot,
                self.inference_runtime_params_config,
                self.inference_runtime_topics_config,
                shm_specs,
                lock,
                control_iter_cond,
                inference_ready_cond,
                stop_event,
                episode_complete_event,
                num_control_iters,
                inference_ready_flag,

                episode_queue_handle_b,
            ),
            daemon=False
        )
        inference_runner.start()
        controller.start()

        try:
            # Robust join loop: timeouts + exitcode monitoring
            procs = [inference_runner, controller]
            while any(p.is_alive() for p in procs):
                for p in procs:
                    p.join(timeout=0.5)
                    # If a child died unexpectedly, request shutdown so waiters wake up
                    if p.exitcode is not None and p.exitcode != 0:
                        stop_event.set()
                        with control_iter_cond:
                            control_iter_cond.notify_all()
        finally:
            stop_event.set()
            try:
                with control_iter_cond:
                    control_iter_cond.notify_all()
            except Exception:
                pass
            for p in (inference_runner, controller):
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=3)
            # Only the parent unlinks; guard against child auto-unlinks/resource_tracker
            for key in shm_specs.keys():
                try:
                    shm_specs[key].close()
                except Exception:
                    pass
                try:
                    shm_specs[key].unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                try:
                    resource_tracker.unregister(shm_specs[key]._name, "shared_memory")
                except Exception:
                    pass