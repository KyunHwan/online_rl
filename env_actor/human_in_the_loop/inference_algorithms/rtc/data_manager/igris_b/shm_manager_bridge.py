"""SharedMemoryManager - Standard class for shared state management in RTC inference.

This module provides a SharedMemoryManager class that replaces the Ray-based
RTCStateActor with direct SharedMemory access. The manager encapsulates:
- SharedMemory block creation and attachment
- Synchronization primitives (RLock, Condition, Event, Value)
- Atomic read/write operations for inference-controller coordination

The SharedMemory blocks are created in the parent process (run_online_rl.py)
and attached to by Ray actors via the attach_from_specs() factory method.
"""
from __future__ import annotations

from ctypes import c_bool
from env_actor.auto.inference_algorithms.rtc.data_manager.utils.utils import ShmArraySpec, attach_shared_ndarray
from multiprocessing import Value, resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING
from env_actor.runtime_settings_configs.igris_b.init_params import (
    INIT_JOINT_LIST,
    INIT_HAND_LIST,
)

import torch
import numpy as np

from ...inference_engine_utils.max_deque import MaxDeque

from multiprocessing.synchronize import Condition as ConditionType
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import RLock as RLockType


class SharedMemoryManager:
    """Manages SharedMemory blocks and synchronization for RTC inference.

    This class is instantiated inside Ray actors to provide direct
    SharedMemory access without Ray actor communication overhead.

    The parent process creates SharedMemory and sync primitives, then passes
    specs and primitives to actors which use attach_from_specs() to create
    their manager instance.

    Attributes:
        robot_obs_history: Shared array for proprioception history (num_robot_obs, state_dim)
        cam_images: Shared array for camera images (num_cams, num_img_obs, 3, H, W)
        action_chunk: Shared array for action chunk (num_queries, action_dim)
    """

    def __init__(
        self,
        shm_dict: dict[str, SharedMemory],
        shm_array_dict: dict[str, SharedMemory],
        lock: RLockType,
        control_iter_cond: ConditionType,
        inference_ready_cond: ConditionType,
        stop_event: EventType,
        episode_complete_event: EventType,
        num_control_iters: Value,
        inference_ready_flag: Value,
        is_creator: bool = False,
    ):
        """Initialize SharedMemoryManager with attached arrays and sync primitives.

        Use attach_from_specs() factory method instead of direct construction.
        """
        # SharedMemory handles (for cleanup)
        self._shm_dict = shm_dict
        self._is_creator = is_creator

        # Shared arrays (direct memory access)
        self._shm_array_dict = shm_array_dict

        # Synchronization primitives
        self._lock = lock
        self._control_iter_cond = control_iter_cond
        self._inference_ready_cond = inference_ready_cond
        self._stop_event = stop_event
        self._episode_complete_event = episode_complete_event
        self._num_control_iters = num_control_iters
        self._inference_ready_flag = inference_ready_flag

        # Delay tracking for guided inference
        self._delay_queue = MaxDeque(buffer_len=5)
        self._delay_queue.add(5)  # Initial delay estimate

    @classmethod
    def attach_from_specs(
        cls,
        shm_specs: dict[str: ShmArraySpec],
        lock: RLockType,
        control_iter_cond: ConditionType,
        inference_ready_cond: ConditionType,
        stop_event: EventType,
        episode_complete_event: EventType,
        num_control_iters: Value,
        inference_ready_flag: Value,
    ) -> SharedMemoryManager:
        """Factory method to create manager by attaching to existing SharedMemory.

        Called by Ray actors to attach to SharedMemory created by parent process.

        Args:
            shm_specs: Dict of ShmArraySpec for SharedMemory blocks
            lock: Shared RLock for atomic operations
            control_iter_cond: Shared Condition for control iteration waits
            inference_ready_cond: Shared Condition for inference ready waits
            stop_event: Shared Event for shutdown signaling
            episode_complete_event: Shared Event for episode completion signaling
            num_control_iters: Shared Value for control iteration counter
            inference_ready_flag: Shared Value for inference ready signal

        Returns:
            SharedMemoryManager instance attached to the shared arrays
        """
        shm_handle_dict, shm_array_dict = attach_shared_ndarray(shm_specs)

        return cls(
            shm_dict=shm_handle_dict,
            shm_array_dict=shm_array_dict,
            lock=lock,
            control_iter_cond=control_iter_cond,
            inference_ready_cond=inference_ready_cond,
            stop_event=stop_event,
            episode_complete_event=episode_complete_event,
            num_control_iters=num_control_iters,
            inference_ready_flag=inference_ready_flag,
            is_creator=False,
        )

    # =========================================================================
    # Synchronization Methods
    # =========================================================================

    def wait_for_min_actions(self, min_actions: int) -> str:
        """Block until control iterations reach threshold or episode completes.

        Used by InferenceActor to wait until sufficient actions have been
        executed before running the next inference, or until the episode ends.

        Args:
            min_actions: Minimum number of control iterations to wait for

        Returns:
            'min_actions' - threshold reached, continue inference
            'episode_complete' - episode ended, handle transition
            'stop' - shutdown requested
        """
        with self._control_iter_cond:
            self._control_iter_cond.wait_for(
                lambda: (
                    self._stop_event.is_set()
                    or self._episode_complete_event.is_set()
                    or self._num_control_iters.value >= min_actions
                )
            )

            if self._stop_event.is_set():
                return 'stop'
            if self._episode_complete_event.is_set():
                return 'episode_complete'
            return 'min_actions'

    def wait_for_inference_ready(self) -> bool:
        """Block until inference actor signals ready.

        Called by ControllerActor before starting the control loop.

        Returns:
            True if inference ready, False if stop_event was set
        """
        with self._inference_ready_cond:
            self._inference_ready_cond.wait_for(
                lambda: self._stop_event.is_set() or self._inference_ready_flag.value
            )
            return not self._stop_event.is_set()

    def set_inference_ready(self) -> None:
        """Signal that inference actor initialization is complete.

        Called by InferenceActor after policy loading and CUDA warmup.
        """
        with self._lock:
            self._inference_ready_flag.value = True
        with self._inference_ready_cond:
            self._inference_ready_cond.notify_all()
    
    def set_inference_not_ready(self) -> None:
        """Signal that inference actor initialization is about to begin.

        Called by InferenceActor before policy loading and CUDA warmup.
        """
        with self._lock:
            self._inference_ready_flag.value = False
        with self._inference_ready_cond:
            self._inference_ready_cond.notify_all()

    def signal_episode_complete(self) -> None:
        """Signal that the episode for loop has completed.

        Called by ControllerActor after the episode for loop finishes.
        """
        self._episode_complete_event.set()
        with self._control_iter_cond:
            self._control_iter_cond.notify_all()

    def is_episode_complete(self) -> bool:
        """Check if episode complete event is set.

        Returns:
            True if the episode complete event is set
        """
        return self._episode_complete_event.is_set()

    def clear_episode_complete(self) -> None:
        """Clear the episode complete event for new episode.

        Called by ControllerActor after wait_for_inference_ready() returns,
        before starting a new episode.
        """
        self._episode_complete_event.clear()

    def notify_step(self) -> None:
        """Wake up any waiting inference actor.

        Called by ControllerActor after each control step to potentially
        wake up the inference actor waiting on wait_for_min_actions().
        """
        with self._control_iter_cond:
            self._control_iter_cond.notify_all()

    # =========================================================================
    # Atomic Read Operations
    # =========================================================================

    def atomic_read_for_inference(self) -> dict:
        """Atomically read all state needed for inference.

        Returns a snapshot of the current state including:
        - proprio: Robot observation history (copy)
        - head: Camera image history (copy)
        - left: Camera image history (copy)
        - right: Camera image history (copy)
        - action: Current action chunk (copy)
        - num_control_iters: Number of control iterations since last inference
        - estimated_delay: Maximum delay from delay queue

        Returns:
            Dict with state snapshot (copies, not references) as numpy array
        """
        cur_state = {}
        with self._lock:
            cur_state['num_control_iters'] = self._num_control_iters.value
            for key in self._shm_array_dict.keys():
                cur_state[key] = self._shm_array_dict[key].copy()
            cur_state['prev_action'] = np.zeros_like(self._shm_array_dict['action'])
            k = max(self._shm_array_dict['action'].shape[0] - self._num_control_iters.value, 0)
            np.copyto(cur_state['prev_action'][:k, :], self._shm_array_dict['action'][self._num_control_iters.value:, :], casting='no')
            cur_state['est_delay'] = self._delay_queue.max()

            return cur_state

    # =========================================================================
    # Atomic Write Operations
    # =========================================================================

    def atomic_write_obs_and_increment_get_action(
        self,
        obs: dict[str, np.ndarray],
        action_chunk_size: int,
    ) -> int:
        """Atomically update observations and increment counter.

        Updates the observation history with new data and increments the
        control iteration counter. History is shifted (FIFO) with new
        observation at index 0.

        Args:
            robot_obs: New robot state (state_dim,)
            cam_images: New camera images (num_img_obs, 3, H, W)

        Returns:
            action
        """
        with self._lock:
            # Increment counter
            self._num_control_iters.value += 1

            # Shift robot history and insert new observation at front
            if self._shm_array_dict['proprio'].shape[0] > 1:
                self._shm_array_dict['proprio'][1:] = self._shm_array_dict['proprio'][:-1]
            for key in obs.keys():
                if key == 'proprio':
                    np.copyto(self._shm_array_dict[key][0], obs[key], casting='no')
                else:
                    np.copyto(self._shm_array_dict[key], obs[key], casting='no')
            action_idx = min(self._num_control_iters.value - 1, action_chunk_size - 1)
            action = self._shm_array_dict['action'][action_idx].copy()
            
            self.notify_step()

            return action

    def write_action_chunk_n_update_iter_val(self, action_chunk: np.ndarray, executed: int) -> None:
        """Write new action chunk and update tracking.

        Called by InferenceActor after computing a new action chunk.
        Updates the action chunk, adds the executed count to delay queue,
        and decrements the control counter.

        Args:
            action_chunk: New action chunk (num_queries, action_dim)
            executed: Number of actions executed since last inference
        """
        with self._lock:
            if len(action_chunk.shape) == 3:
                action_chunk = action_chunk.squeeze(0) 
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()
            np.copyto(self._shm_array_dict['action'], action_chunk.astype(np.float32, copy=False), casting='no')
            self._num_control_iters.value = self._num_control_iters.value - executed
            self._delay_queue.add(self._num_control_iters.value)

    def init_action_chunk(self) -> None:
        """Initialize action chunk with a specific value.

        Args:
            init_chunk: Initial action chunk (num_queries, action_dim)
        """
        with self._lock:
            """ Serve init action for RTC Guided Inference """
            init_vec = np.asarray(
                INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6] + INIT_HAND_LIST[:6] + INIT_HAND_LIST[6:],
                dtype=np.float32,
            )
            # Convert joints to radians, scale fingers
            init_vec[:12] *= np.pi / 180.0
            init_vec[12:] *= 0.03

            # Repeat across all rows
            np.copyto(self._shm_array_dict['action'], 
                      np.tile(init_vec, (self._shm_array_dict['action'].shape[0], 1)))

    def bootstrap_obs_history(
        self,
        obs_history
    ) -> None:
        """Bootstrap observation history with initial state.

        Fills the entire history with the initial observation (repeated).

        Args:
            obs_history: 
                Dictionary of robot proprio state and camera images (head, left, right)
            proprio: Initial robot state (state_dim,)
            images: Initial camera images (num_cams, 3, H, W) - single frame
        """
        with self._lock:
            # Repeat initial state across history
            for key in obs_history.keys():
                if key == 'proprio':
                    np.copyto(self._shm_array_dict['proprio'], 
                      np.repeat(obs_history['proprio'].reshape(1, -1),
                        self._shm_array_dict['proprio'].shape[0],
                        axis=0,),
                      casting='no')
                else:
                    np.copyto(self._shm_array_dict[key], obs_history[key], casting='no')

    def reset(self) -> None:
        """Reset state for new episode.

        Resets the control counter and clears the delay queue.
        Action chunk and observation history are preserved.
        """
        with self._lock:
            self._num_control_iters.value = 0
            self._delay_queue.clear()
            self._delay_queue.add(5)  # Re-initialize with default delay

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def stop_event_is_set(self) -> bool:
        return self._stop_event.is_set()

    def signal_stop(self) -> None:
        """Signal stop event and wake all waiters."""
        self._stop_event.set()
        # Notify both conditions to wake up all waiters
        try:
            with self._control_iter_cond:
                self._control_iter_cond.notify_all()
        except Exception:
            pass
        try:
            with self._inference_ready_cond:
                self._inference_ready_cond.notify_all()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Close SharedMemory handles.

        If is_creator, also unlinks the SharedMemory (parent process only).
        """
        for key, shm in self._shm_dict.items():
            try:
                shm.close()
            except Exception:
                pass
            if self._is_creator:
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
