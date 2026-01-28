import os
import sys
import argparse

import torch

import ray
from ray.util.queue import Queue as RayQueue
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from env_actor.control_actor import ControllerActor
from env_actor.inference_actor import InferenceActor
from data_bridge.replay_buffer import ReplayBufferActor
from data_bridge.policy_state_manager import PolicyStateManagerActor

from trainer.trainer.online_trainer import train_func

import numpy as np

robot_obs_history_dtype = np.float32
cam_images_dtype  = np.uint8
action_chunk_dtype = np.float32

@ray.remote(num_gpus=2)
def run_training(train_config_path: str):
    """Run TorchTrainer.fit() in a Ray worker process so the GUI thread stays free."""
    print("Running TorchTrainer...")
    # Distributed training
    dist_train_setting_config = ScalingConfig(
        num_workers=2,
        use_gpu=True,
    )
    online_trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config_path,
        scaling_config=dist_train_setting_config,
    )
    return online_trainer.fit()




def start_online_rl(train_config_path, inference_config_path, human_reward_labeler):
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"working_dir": os.getcwd()},
             namespace="online_rl")
    
    # Queue is the bridge between the controller and the reward labeler
    # via which the data reaches the replay buffer.
    episode_queue = RayQueue(maxsize=10) # This sets the maxsize of the Queue to be 10 elements

    policy_state_manager = PolicyStateManagerActor.options(name="policy_state_manager").remote()
    replay_buffer = ReplayBufferActor.options(name="replay_buffer").remote()

    # Environment Actor
    # Create shared memory 
    shared_memory_names = []
    inference_engine = InferenceActor.remote(policy_state_manager_handle=policy_state_manager,
                                             inference_config_path=inference_config_path,
                                             shared_memory_names=shared_memory_names)
    controller = ControllerActor.remote(episode_queue_handle=episode_queue,
                                        inference_config_path=inference_config_path,
                                        shared_memory_names=shared_memory_names)
    
    # Start the environment actor to collect data
    inference_engine.start.remote()
    controller.start.remote()

    train_ref = run_training.remote(train_config_path)

    # Start reward labeler
    if not human_reward_labeler:
        from data_labeler.auto.auto_reward_labeler import AutoRewardLabelerActor as RewardLabeler
    else:
        from data_labeler.human_in_the_loop.hil_reward_labeler import ManualRewardLabelerActor as RewardLabeler
    reward_labeler = RewardLabeler.remote(episode_queue_handle=episode_queue, 
                                            replay_buffer_actor=replay_buffer,
                                            img_frame_key='cam_head',
                                            reward_key='reward')
    _ = reward_labeler.start.remote()
    _ = ray.get(train_ref)

    ray.shutdown()
    sys.exit()

    
    
    

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="Parse for train config and inference_config .yaml files")
    parser.add_argument("--train_config", help="absolute path to the train config .yaml file.", required=True)
    parser.add_argument("--inference_config", help="absolute path to the inference config .yaml file.", required=True)
    parser.add_argument("--human_reward_labeler", action="store_true", help="whether reward labeling is done by a human", default=True)
    args = parser.parse_args()
    if args.train_config:
        args.train_config = os.path.abspath(args.train_config)

    start_online_rl(args.train_config, args.inference_config, args.human_reward_labeler)