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

RAYQUEUE_MAXSIZE = 25

@ray.remote()
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
                    ckpt_dir,
                    default_prompt
                    ):
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="auto",
             namespace="online_rl")
    
    try:
        # Queue is the bridge between the controller and the reward labeler
        # via which the data reaches the replay buffer.
        episode_queue = RayQueue(maxsize=RAYQUEUE_MAXSIZE) # This sets the maxsize of the Queue to be 10 elements

        policy_state_manager = PolicyStateManagerActor.options(resources={"training_pc": 1},
                                                            name="policy_state_manager").remote()
        replay_buffer = ReplayBufferActor.options(resources={"training_pc": 1},
                                                name="replay_buffer").remote()

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
            from env_actor.auto.inference_algorithms.rtc.rtc_actor_openpi import RTCActorOpenpi

            env_actor = RTCActorOpenpi.\
                options(resources={"inference_pc": 1}, num_cpus=4, num_gpus=1).\
                remote(
                    robot=robot,
                    inference_runtime_params_config=inference_runtime_params_config,
                    inference_runtime_topics_config=inference_runtime_topics_config,
                    min_num_actions_executed=30,

                    episode_queue_handle=episode_queue,

                    ckpt_dir=ckpt_dir,
                    default_prompt=default_prompt,
                )
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

        _ = ray.get(train_ref)

    finally:
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
    parser.add_argument("--inference_algorithm", default="rtc", choices=["sequential", "rtc"],
                        help="inference algorithm: 'sequential' or 'rtc' (real-time action chunking)")
    parser.add_argument(
        "--ckpt_dir", "-C", type=str,
        default="/home/robros/Projects/robros_vla_inference_engine/openpi_film/checkpoints/pi05_igris/pi05_igris_b_pnp_v3.3.2/film_15000",
        help="Path to OpenPI checkpoint step directory (contains model.safetensors + assets/)",
    )
    parser.add_argument(
        "--default_prompt", type=str,
        default="Pick up objects on the table and place them into the box.",
        help="Default language prompt for the policy (e.g., 'pick and place')",
    )
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
        args.ckpt_dir,
        args.default_prompt
    )
