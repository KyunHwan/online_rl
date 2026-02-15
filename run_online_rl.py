import os
import sys
import argparse
import json
from ctypes import c_bool
from multiprocessing import Condition, Event, RLock, Value

import torch

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        print(f"WARNING: Could not set multiprocessing start method: {e}")
        print(f"Current method: {torch.multiprocessing.get_start_method()}")

import ray
from ray.util.queue import Queue as RayQueue
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from data_bridge.replay_buffer import ReplayBufferActor
from data_bridge.policy_state_manager import PolicyStateManagerActor

from trainer.trainer.online_trainer import train_func

import numpy as np

robot_obs_history_dtype = np.float32
cam_images_dtype = np.uint8
action_chunk_dtype = np.float32

RAYQUEUE_MAXSIZE=25

@ray.remote#num_gpus=4)
def run_training(train_config_path: str):
    """Run TorchTrainer.fit() in a Ray worker process so the GUI thread stays free."""
    print("Running TorchTrainer...")
    # Distributed training
    dist_train_setting_config = ScalingConfig(
        num_workers=4,
        use_gpu=True,
    )
    online_trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config_path,
        scaling_config=dist_train_setting_config,
    )
    return online_trainer.fit()





def start_online_rl(train_config_path, policy_yaml_path, robot, human_reward_labeler, inference_runtime_params_config, inference_runtime_topics_config=None, inference_algorithm='sequential', ):
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="auto",
             namespace="online_rl")
    
    # Queue is the bridge between the controller and the reward labeler
    # via which the data reaches the replay buffer.
    episode_queue = RayQueue(maxsize=RAYQUEUE_MAXSIZE) # This sets the maxsize of the Queue to be 10 elements

    policy_state_manager = PolicyStateManagerActor.options(resources={"training_pc": 1},
                                                           name="policy_state_manager").remote()
    replay_buffer = ReplayBufferActor.options(resources={"training_pc": 1},
                                              name="replay_buffer").remote(slice_len=80)

    # Environment Actor
    # Load RuntimeParams to get dimensions for SharedMemory
    if robot == "igris_b":
        from env_actor.runtime_settings_configs.igris_b.inference_runtime_params import RuntimeParams
    elif robot == "igris_c":
        from env_actor.runtime_settings_configs.igris_c.inference_runtime_params import RuntimeParams
    else:
        raise ValueError(f"Unknown robot: {robot}")
    
    if isinstance(inference_runtime_params_config, str):
        with open(inference_runtime_params_config, 'r') as f:
            inference_runtime_params_config = json.load(f)
    runtime_params = RuntimeParams(inference_runtime_params_config)

    if isinstance(inference_runtime_topics_config, str):
        with open(inference_runtime_topics_config, 'r') as f:
            inference_runtime_topics_config = json.load(f)

    # RTC (Real-Time Action Chunking)
    if inference_algorithm == 'rtc':
        from env_actor.auto.inference_algorithms.rtc.control_actor import ControllerActor as RTCControllerActor
        from env_actor.auto.inference_algorithms.rtc.inference_actor import InferenceActor as RTCInferenceActor
        from env_actor.auto.inference_algorithms.rtc.data_manager.utils.utils import create_shared_ndarray


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
        lock = RLock()
        control_iter_cond = Condition(lock)      # For num_control_iters waits
        inference_ready_cond = Condition(lock)   # For inference_ready waits
        stop_event = Event()
        episode_complete_event = Event()         # For episode completion signaling
        num_control_iters = Value('i', 0, lock=False)
        inference_ready_flag = Value(c_bool, False, lock=False)

        # Create inference actor (GPU-resident) with SharedMemory specs
        inference_engine = RTCInferenceActor.\
                        options(resources={"inference_pc": 1}).\
                        remote(
                            runtime_params=runtime_params,
                            policy_yaml_path=policy_yaml_path,
                            policy_state_manager_handle=policy_state_manager,
                            shm_specs=shm_specs,
                            lock=lock,
                            control_iter_cond=control_iter_cond,
                            inference_ready_cond=inference_ready_cond,
                            stop_event=stop_event,
                            episode_complete_event=episode_complete_event,
                            num_control_iters=num_control_iters,
                            inference_ready_flag=inference_ready_flag,
                        )

        # Create controller actor (Robot I/O) with SharedMemory specs
        controller = RTCControllerActor.\
                        options(resources={"inference_pc": 1}).\
                        remote(
                            runtime_params=runtime_params,
                            inference_runtime_topics_config=inference_runtime_topics_config,
                            robot=robot,
                            episode_queue_handle=episode_queue,
                            shm_specs=shm_specs,
                            lock=lock,
                            control_iter_cond=control_iter_cond,
                            inference_ready_cond=inference_ready_cond,
                            stop_event=stop_event,
                            episode_complete_event=episode_complete_event,
                            num_control_iters=num_control_iters,
                            inference_ready_flag=inference_ready_flag,
                        )

        # Start the RTC actors
        inference_engine.start.remote()
        controller.start.remote()
    else:
        # Sequential inference
        from env_actor.auto.inference_algorithms.sequential.sequential_actor import SequentialActor
        env_actor = SequentialActor.\
                        options(resources={"inference_pc": 1}).\
                        remote(
                            runtime_params=runtime_params,
                            inference_runtime_topics_config=inference_runtime_topics_config,
                            robot=robot,
                            policy_yaml_path=policy_yaml_path,
                            policy_state_manager_handle=policy_state_manager,
                            episode_queue_handle=episode_queue,
                        )
        env_actor.start.remote()

    train_ref = run_training.\
                    options(resources={"training_pc": 1}).\
                    remote(train_config_path)

    # Start reward labeler
    if not human_reward_labeler:
        from data_labeler.auto.auto_reward_labeler import AutoRewardLabelerActor as RewardLabeler
    else:
        from data_labeler.human_in_the_loop.hil_reward_labeler import ManualRewardLabelerActor as RewardLabeler
    reward_labeler = RewardLabeler.\
                        options(resources={"labeling_pc": 1}).\
                        remote(episode_queue_handle=episode_queue,
                               replay_buffer_actor=replay_buffer,
                               img_frame_key='head',
                               reward_key='reward')
    _ = reward_labeler.start.remote()

    try:
        _ = ray.get(train_ref)
    finally:
        # Cleanup for RTC: signal stop and cleanup SharedMemory
        if inference_algorithm == 'rtc':
            from multiprocessing import resource_tracker

            # Signal actors to stop
            stop_event.set()
            episode_complete_event.set()  # Wake any waiters on episode_complete
            # Notify both conditions to wake up all waiters
            try:
                with control_iter_cond:
                    control_iter_cond.notify_all()
            except Exception:
                pass
            try:
                with inference_ready_cond:
                    inference_ready_cond.notify_all()
            except Exception:
                pass

            # Cleanup SharedMemory (parent process is responsible for unlinking)
            for shm in (rob_shm, head_cam_shm, left_cam_shm, right_cam_shm, act_shm):
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                try:
                    resource_tracker.unregister(shm._name, "shared_memory")
                except Exception:
                    pass

    ray.shutdown()
    sys.exit()

    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse for train config and inference_config .yaml files")
    parser.add_argument("--train_config", help="absolute path to the train config .yaml file.", required=True)
    parser.add_argument("--policy_yaml", help="absolute path to the policy config .yaml file.", required=True)
    parser.add_argument("--robot", help="igris_b or igris_c", required=True)
    parser.add_argument("--human_reward_labeler", action="store_true", help="whether reward labeling is done by a human")
    parser.add_argument("--inference_runtime_params_config", help="absolute path to the inference runtime params config file.", required=True)
    parser.add_argument("--inference_runtime_topics_config", help="absolute path to the inference runtime topics config file.", required=True)
    parser.add_argument("--inference_algorithm", default="sequential", choices=["sequential", "rtc"],
                        help="inference algorithm: 'sequential' or 'rtc' (real-time action chunking)")
    args = parser.parse_args()
    if args.train_config:
        args.train_config = os.path.abspath(args.train_config)

    start_online_rl(
        args.train_config,
        args.policy_yaml,
        args.robot,
        args.human_reward_labeler,
        args.inference_runtime_params_config,
        args.inference_runtime_topics_config,
        args.inference_algorithm,
    )
