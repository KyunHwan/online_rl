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
from data_bridge.state_manager import StateManagerActor

from trainer.trainer.online_trainer import train_func

import numpy as np

robot_obs_history_dtype = np.float32
cam_images_dtype = np.uint8
action_chunk_dtype = np.float32

RAYQUEUE_MAXSIZE = 25

@ray.remote
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





def start_online_rl(train_config_path, 
                    policy_yaml_path, 
                    robot, 
                    human_reward_labeler, 
                    inference_runtime_params_config, 
                    inference_runtime_topics_config, 
                    inference_algorithm,
                    ):
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="auto",
             namespace="online_rl",
             runtime_env={"working_dir": "."})
    
    try:
        # Queue is the bridge between the controller and the reward labeler
        # via which the data reaches the replay buffer.
        episode_queue = RayQueue(maxsize=RAYQUEUE_MAXSIZE) # This sets the maxsize of the Queue to be 10 elements

        policy_state_manager = StateManagerActor.options(resources={"training_pc": 1},
                                                            name="policy_state_manager").remote()
        # norm_stats_state_manager = StateManagerActor.options(resources={"training_pc": 1},
        #                                                    name="norm_stats_state_manager").remote()
        replay_buffer = ReplayBufferActor.options(resources={"training_pc": 1},
                                                name="replay_buffer").remote(slice_len=80)

        # Environment Actor
        # Load RuntimeParams to get dimensions for SharedMemory
        if robot == "igris_b":
            from env_actor.runtime_settings_configs.robots.igris_b.inference_runtime_params import RuntimeParams
        elif robot == "igris_c":
            from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams
        else:
            raise ValueError(f"Unknown robot: {robot}")

        # RTC (Real-Time Action Chunking)
        if inference_algorithm == 'rtc':
            from env_actor.auto.inference_algorithms.rtc.rtc_actor import RTCActor

            env_actor = RTCActor.\
                options(resources={"inference_pc": 1}).\
                remote(
                    robot=robot,
                    policy_yaml_path=policy_yaml_path,
                    inference_runtime_params_config=inference_runtime_params_config,
                    inference_runtime_topics_config=inference_runtime_topics_config,

                    episode_queue_handle=episode_queue,
                )
        else:
            # Sequential inference
            from env_actor.auto.inference_algorithms.sequential.sequential_actor import SequentialActor
            env_actor = SequentialActor.\
                            options(resources={"inference_pc": 1}, num_cpus=4, num_gpus=1).\
                            remote(
                                inference_runtime_params_config=inference_runtime_params_config,
                                inference_runtime_topics_config=inference_runtime_topics_config,
                                robot=robot,
                                policy_yaml_path=policy_yaml_path,
                                policy_state_manager_handle=policy_state_manager,
                                episode_queue_handle=episode_queue,
                            )
        print("running env_actor...")
        env_actor.start.remote()

        print("running training...")
        train_ref = run_training.\
                        options(resources={"training_pc": 1}).\
                        remote(train_config_path)

        # Start reward labeler
        if not human_reward_labeler:
            from data_labeler.auto.auto_reward_labeler import AutoRewardLabelerActor as RewardLabeler
        else:
            from data_labeler.human_in_the_loop.hil_reward_labeler import ManualRewardLabelerActor as RewardLabeler

        print("running labeler...")
        reward_labeler = RewardLabeler.\
                            options(resources={"labeling_pc": 1}).\
                            remote(episode_queue_handle=episode_queue,
                                replay_buffer_actor=replay_buffer,
                                img_frame_key='head',
                                reward_key='reward')
        _ = reward_labeler.start.remote()

        _ = ray.get(train_ref)

    except Exception as e:
        print(f"Online RL failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        ray.shutdown()

    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse for train config and inference_config .yaml files")
    parser.add_argument("--robot", 
                        required=True,
                        help="igris_b or igris_c")
    parser.add_argument("--human_reward_labeler", 
                        action="store_true", 
                        help="whether reward labeling is done by a human")
    parser.add_argument("--train_config", 
                        default="/home/user/Projects/online_rl/trainer/experiment_training/imitation_learning/vfp_single_expert/exp2/vfp_single_expert_depth.yaml",
                        help="absolute path to the train config .yaml file.")
    parser.add_argument("--policy_yaml", 
                        default="./env_actor/policy/policies/openpi_policy/openpi_policy.yaml",
                        help="path to the policy config .yaml file.")
    parser.add_argument("--inference_runtime_params_config", 
                        default="/home/robros/Projects/online_rl/env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json",
                        help="absolute path to the inference runtime params config file.")
    parser.add_argument("--inference_runtime_topics_config", 
                        default="/home/robros/Projects/online_rl/env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json",
                        help="absolute path to the inference runtime topics config file.")
    parser.add_argument("--inference_algorithm", 
                        default="rtc", 
                        choices=["sequential", "rtc"],
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
