from env_actor.auto.inference_algorithms.rtc.data_manager.utils.utils import ShmArraySpec
from typing import TYPE_CHECKING
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Value

import numpy as np

from multiprocessing.synchronize import Condition as ConditionType
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import RLock as RLockType



class SharedMemoryInterface:
    def __init__(self, 
                 robot: str,
                 shm_specs: dict[str, SharedMemory],
                 lock: RLockType,
                 control_iter_cond: ConditionType,
                 inference_ready_cond: ConditionType,
                 stop_event: EventType,
                 episode_complete_event: EventType,
                 num_control_iters: Value,
                 inference_ready_flag: Value,
                 is_creator: bool = False,):
        if robot == "igris_b":
            from env_actor.auto.inference_algorithms.rtc.data_manager.igris_b.shm_manager_bridge import SharedMemoryManager
        elif robot == "igris_c":
            from env_actor.auto.inference_algorithms.rtc.data_manager.igris_c.shm_manager_bridge import SharedMemoryManager
        self.shm_manager = SharedMemoryManager.attach_from_specs(
            shm_specs=shm_specs,
            lock=lock,
            control_iter_cond=control_iter_cond,
            inference_ready_cond=inference_ready_cond,
            stop_event=stop_event,
            episode_complete_event=episode_complete_event,
            num_control_iters=num_control_iters,
            inference_ready_flag=inference_ready_flag,
        )

    @classmethod
    def attach_from_specs(
        cls,
        robot: str,
        shm_specs: dict[str: ShmArraySpec],
        lock: RLockType,
        control_iter_cond: ConditionType,
        inference_ready_cond: ConditionType,
        stop_event: EventType,
        episode_complete_event: EventType,
        num_control_iters: Value,
        inference_ready_flag: Value,
    ) -> SharedMemoryInterface:
        return cls(
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
    
    # =========================================================================
    # Synchronization Methods
    # =========================================================================
    def wait_for_min_actions(self, min_actions: int) -> str:
        return self.shm_manager.wait_for_min_actions(min_actions)

    def wait_for_inference_ready(self) -> bool:
        return self.shm_manager.wait_for_inference_ready()
    
    def set_inference_ready(self) -> None:
        self.shm_manager.set_inference_ready()
    
    def set_inference_not_ready(self) -> None:
        self.shm_manager.set_inference_not_ready()

    def signal_episode_complete(self) -> None:
        self.shm_manager.signal_episode_complete()
        
    def is_episode_complete(self) -> bool:
        return self.shm_manager.is_episode_complete()

    def clear_episode_complete(self) -> None:
        self.shm_manager.clear_episode_complete()

    def notify_step(self) -> None:
        self.shm_manager.notify_step()

    # =========================================================================
    # Atomic Read Operations
    # =========================================================================

    def atomic_read_for_inference(self) -> dict:
        return self.shm_manager.atomic_read_for_inference()
    
    # =========================================================================
    # Atomic Write Operations
    # =========================================================================

    def atomic_write_obs_and_increment_get_action(
        self,
        obs: dict[str, np.ndarray],
        action_chunk_size: int,
    ) -> int:
        return self.shm_manager.atomic_write_obs_and_increment_get_action(obs=obs, 
                                                                   action_chunk_size=action_chunk_size)
        
    def write_action_chunk_n_update_iter_val(self, action_chunk: np.ndarray, executed: int) -> None:
        self.shm_manager.write_action_chunk_n_update_iter_val(action_chunk=action_chunk, executed=executed)
    
    def init_action_chunk(self) -> None:
        self.shm_manager.init_action_chunk()

    def bootstrap_obs_history(
        self,
        obs_history
    ) -> None:
        self.shm_manager.bootstrap_obs_history(obs_history=obs_history)
        
    def reset(self) -> None:
        self.shm_manager.reset()

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def stop_event_is_set(self) -> bool:
        return self.shm_manager.stop_event_is_set()
    
    def signal_stop(self) -> None:
        self.shm_manager.signal_stop()

    def cleanup(self) -> None:
        self.shm_manager.cleanup()