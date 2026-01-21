import os
import argparse

import torch

import ray
from ray.util.queue import Queue as RayQueue
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from env_actor.control_actor import ControllerActor
from env_actor.inference_actor import InferenceActor
from data_labeler.reward_labeler import RewardLabelerActor
from data_bridge.replay_buffer import ReplayBufferActor
from data_bridge.policy_state_manager import PolicyStateManagerActor

from trainer.online_trainer import train_func



def start_online_rl(train_config_path, inference_config_path):
    
    # Queue is the bridge between the controller and the reward labeler
    # via which the data reaches the replay buffer.
    episode_queue = RayQueue(maxsize=10) # This sets the maxsize of the Queue to be 10 Gb

    policy_state_manager = PolicyStateManagerActor.options(name="policy_state_manager").remote()
    replay_buffer = ReplayBufferActor.options(name="replay_buffer").remote()
    reward_labeler = RewardLabelerActor.remote(episode_queue_handle=episode_queue, 
                                               replay_buffer_handle=replay_buffer)

    # Environment Actor
    shared_memory_names = []
    inference_engine = InferenceActor.remote(policy_state_manager_handle=policy_state_manager,
                                             inference_config_path=inference_config_path,
                                             shared_memory_names=shared_memory_names)
    controller = ControllerActor.remote(episode_queue_handle=episode_queue,
                                        inference_config_path=inference_config_path,
                                        shared_memory_names=shared_memory_names)
    
    # Distributed training
    dist_train_setting_config = ScalingConfig(
        num_workers=2,
        use_gpu=True,
    )
    print("trying to run TorchTrainer")
    online_trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config_path,
        scaling_config=dist_train_setting_config,
    )

    # Start the environment actor to collect data
    inference_engine.start.remote()
    controller.start.remote()

    # Start reward labeler
    reward_labeler.start.remote()

    # Start distributed training 
    online_trainer.fit()

    

    
    

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    ray.init(runtime_env={"working_dir": os.getcwd()})

    parser = argparse.ArgumentParser(description="Parse for train config and inference_config .yaml files")
    parser.add_argument("--train_config", help="absolute path to the train config .yaml file.", required=True)
    parser.add_argument("--inference_config", help="absolute path to the inference config .yaml file.", required=True)
    args = parser.parse_args()
    if args.train_config:
        args.train_config = os.path.abspath(args.train_config)
    start_online_rl(args.train_config, args.inference_config)
    
    ray.shutdown()