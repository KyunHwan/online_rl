#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief   record_episode (async save version)
@modified  Async save with queue support
"""

import os
import sys
import time
import json
import h5py
import cv2
import argparse
import signal
import numpy as np
import gc
import psutil
from tqdm import tqdm
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import wave
import rclpy
from rclpy.executors import SingleThreadedExecutor
import std_msgs.msg
import threading

from utils.data_dict import GenericRecorder, load_config
from utils.camera_utils import RBRSCamera
import json, numpy as np

import dynamixel_sdk
import pyttsx3
import subprocess

MAX_SAVE_QUEUE_SIZE = 8

portHandler = dynamixel_sdk.PortHandler("/dev/igris_head")
packetHandler = dynamixel_sdk.PacketHandler(2.0)
labels_string = "{\"version\": 1, \"series\": [{\"id\": \"rewards\", \"display_name\": \"Rewards\", \"path\": \"/labels/rewards\", \"dtype\": \"float32\", \"kind\": \"timeseries\"}, {\"id\": \"task_done\", \"display_name\": \"Task Done\", \"path\": \"/labels/task_done\", \"dtype\": \"bool\", \"kind\": \"timeseries\"}], \"tasks\": [], \"task_tree\": {\"id\": \"pick_and_place\", \"display_name\": \"pick_and_place\", \"path\": \"/labels/task\", \"status_path\": \"/labels/task/status\", \"children\": []}}"
def deg2dxl(angle_deg, resolution=4095):
    value = int((angle_deg % 360) / 360 * resolution)
    byte0 = value & 0xFF
    byte1 = (value >> 8) & 0xFF
    byte2 = (value >> 16) & 0xFF
    byte3 = (value >> 24) & 0xFF

    return [byte0, byte1, byte2, byte3]

def set_head_pan(pan_angle):
    DXL_ID = 2
    ADDR_GOAL_POSITION = 116
    LEN_GOAL_POSITION = 4
    data = deg2dxl(pan_angle)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, int.from_bytes(bytearray(data), byteorder='little'))
    if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
        print(f"{RED}Failed to set head pan: {packetHandler.getTxRxResult(dxl_comm_result)}{RESET}")

def set_head_tilt(tilt_angle):
    DXL_ID = 1
    ADDR_GOAL_POSITION = 116
    LEN_GOAL_POSITION = 4
    data = deg2dxl(tilt_angle)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, int.from_bytes(bytearray(data), byteorder='little'))
    if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
        print(f"{RED}Failed to set head tilt: {packetHandler.getTxRxResult(dxl_comm_result)}{RESET}")

def set_head_torque(enable):
    DXL_IDs = [1, 2]
    ADDR_TORQUE_ENABLE = 64
    for DXL_ID in DXL_IDs:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 1 if enable else 0)
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print(f"{RED}Failed to set head torque for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}{RESET}")

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"

def generate_buzzer_sound(
    freq=1000,
    duration=0.15,
    silence_duration=0.1,   # 앞부분 무음 (초)
    filename="/tmp/buzzer.wav"
):
    sample_rate = 44100

    silence = np.zeros(int(sample_rate * silence_duration), dtype=np.int16)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * freq * t)
    tone_audio = (tone * 32767 * 0.8).astype(np.int16)

    audio = np.concatenate([silence, tone_audio])

    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    return filename

# Generate buzzer sounds at module load
BUZZER_START = "/tmp/buzzer_start.wav"
BUZZER_STOP = "/tmp/buzzer_stop.wav"
generate_buzzer_sound(freq=1000, duration=0.15,silence_duration=0.05, filename=BUZZER_START)
generate_buzzer_sound(freq=600, duration=0.2,silence_duration=0.0, filename=BUZZER_STOP)


def play_beep(sound_type="start"):
    sounds = {
        "start": BUZZER_START,
        "stop": BUZZER_STOP,
        "bell": "/usr/share/sounds/freedesktop/stereo/bell.oga",  # for preload/dummy
    }
    sound_file = sounds.get(sound_type, sounds["start"])
    try:
        subprocess.run(["paplay", sound_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"{YELLOW}Warning: Could not play beep: {e}{RESET}")

    
@dataclass
class SaveTask:
    data_dict: Dict[str, Any]
    meta: Dict[str, Any]
    config: Dict[str, Any]
    config_path: str
    last_step: int
    task_state: str
    current_tilt: float
    episode_index: int
    task_name: str
    image_compress: bool
    encode_param: Any

class SaveWorker:
    def __init__(self):
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.pending_count = 0
        self.lock = threading.Lock()

    def _worker(self):
        while self.running:
            try:
                task = self.queue.get(timeout=1.0)
                if task is None:
                    continue
                self._save_episode(task)
                with self.lock:
                    self.pending_count -= 1
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"{RED}SaveWorker error: {e}{RESET}")
                import traceback
                traceback.print_exc()

    def submit(self, task: SaveTask) -> bool:
        with self.lock:
            if self.pending_count >= MAX_SAVE_QUEUE_SIZE:
                print(f"{YELLOW}Warning: Save queue full ({self.pending_count} pending). Waiting...{RESET}")
                return False
            self.pending_count += 1
        self.queue.put(task)
        print(f"{GREEN}Episode queued for saving (pending: {self.pending_count}){RESET}")
        return True

    def get_pending_count(self) -> int:
        with self.lock:
            return self.pending_count

    def wait_for_slot(self):
        while self.get_pending_count() >= MAX_SAVE_QUEUE_SIZE:
            print(f"{YELLOW}Waiting for save slot...{RESET}")
            time.sleep(0.5)

    def shutdown(self):
        self.running = False
        self.queue.join()
        self.thread.join(timeout=5.0)

    def _save_episode(self, task: SaveTask):
        print(f"{GREEN}Starting background save for episode {task.episode_index}...{RESET}")
        start_save_time = time.time()

        data_dict = task.data_dict
        meta = task.meta
        config = task.config
        last_step = task.last_step
        image_compress = task.image_compress
        encode_param = task.encode_param

        compressed_image_len = []

        if image_compress:
            print(f"{GREEN}Processing compressed images...{RESET}")

            for k in list(data_dict.keys()):
                if "image" not in k:
                    continue

                imgs = data_dict[k]
                print("=== Debug: data_dict[{}] summary ===".format(k))
                print(f"Total image num: {len(imgs)}")

                none_indices = [i for i, img in enumerate(imgs) if img is None]
                print(f"None image num: {len(none_indices)}")
                if none_indices:
                    print(f"None index example: {none_indices[:20]}")

                comp_list = []
                comp_lens = []

                for i, img_data in enumerate(imgs):
                    try:
                        if img_data is None:
                            comp_list.append(None)
                            comp_lens.append(0)
                            continue

                        if isinstance(img_data, np.ndarray) and img_data.ndim == 1:
                            comp_list.append(img_data)
                            comp_lens.append(len(img_data))
                            continue

                        if not isinstance(img_data, np.ndarray) or img_data.ndim == 1:
                            comp_list.append(img_data)
                            comp_lens.append(len(img_data) if img_data is not None else 0)
                            continue

                        img = img_data
                        if isinstance(img, np.ndarray) and img.ndim > 1:
                            _, enc = cv2.imencode(".jpg", img, encode_param)
                            enc = enc.ravel()
                            comp_list.append(enc)
                            comp_lens.append(len(enc))
                        else:
                            print(f"{RED}Error: Cannot convert image data to numpy array (type: {type(img)}){RESET}")
                            comp_list.append(None)
                            comp_lens.append(0)
                            continue

                        del img, img_data
                        if i % 50 == 0:
                            gc.collect()

                    except Exception as e:
                        print(f"{RED}Error encoding image at index {i}: {e}{RESET}")
                        comp_list.append(None)
                        comp_lens.append(0)
                        continue

                data_dict[k] = comp_list
                compressed_image_len.append(comp_lens)
                del imgs
                gc.collect()

        print(f"{GREEN}Padding compressed images...{RESET}")
        gc.collect()
        if compressed_image_len:
            max_len = 0
            max_timesteps = 0
            for camera_lengths in compressed_image_len:
                if camera_lengths:
                    max_len = max(max_len, max(camera_lengths))
                    max_timesteps = max(max_timesteps, len(camera_lengths))

            if max_len > 0 and max_timesteps > 0:
                padded_compressed_image_len = np.zeros((len(compressed_image_len), max_timesteps), dtype=np.int32)

                for i, camera_lengths in enumerate(compressed_image_len):
                    if camera_lengths:
                        for j, length in enumerate(camera_lengths):
                            if j < max_timesteps:
                                padded_compressed_image_len[i, j] = length

                compressed_image_len = padded_compressed_image_len

                for k in list(data_dict.keys()):
                    if "image" in k:
                        compressed_image_list = data_dict[k]
                        padded_compressed_image_list = []

                        for compressed_image in compressed_image_list:
                            if compressed_image is None:
                                padded_compressed_image = np.zeros(max_len, dtype='uint8')
                            else:
                                padded_compressed_image = np.zeros(max_len, dtype='uint8')
                                image_len = len(compressed_image)
                                padded_compressed_image[:image_len] = compressed_image
                            padded_compressed_image_list.append(padded_compressed_image)

                        data_dict[k] = padded_compressed_image_list
                        del compressed_image_list, padded_compressed_image_list
                del padded_compressed_image_len
            gc.collect()

        if task.task_state == "start" or task.task_state == "stop":
            meta['/metadata/data_collection_error'] = False
            dataset_dir = os.path.join("/home/robros/Fixer/dataset")
        elif task.task_state == "mark":
            meta['/metadata/data_collection_error'] = True
            dataset_dir = os.path.join("/home/robros/Fixer/dataset", "corrupted")

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        episode_index_str = f"{task.episode_index:04d}"
        timestamp_str = time.strftime("%y%m%d%H%M", time.localtime())
        dataset_name = f"test_{task.task_name}_episode_{episode_index_str}_{timestamp_str}"
        dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")

        print(f"Dataset path: {dataset_path}")

        T = last_step

        if "/labels/task_done" in data_dict:
            cur = data_dict["/labels/task_done"]
            cur_len = len(cur)
            if cur_len < T:
                cur.extend([False] * (T - cur_len))
            elif cur_len > T:
                data_dict["/labels/task_done"] = cur[:T]
        else:
            data_dict["/labels/task_done"] = [False] * T

        with h5py.File(dataset_path, "w") as f:
            obs_grp  = f.create_group("observation")
            act_grp  = f.create_group("action")
            lbl_grp  = f.create_group("labels")
            meta_grp = f.create_group("metadata")

            for k, v in data_dict.items():
                if not k.startswith("/observation/"):
                    continue
                name = k.replace("/observation/", "")

                try:
                    if "image" in k:
                        if not v:
                            print(f"{RED}Warning: Skipping empty image dataset {k}{RESET}")
                            continue

                        arr = np.array(v, dtype=np.uint8)
                        chunk_size = (1, arr.shape[1]) if arr.ndim == 2 else None
                        if chunk_size is not None:
                            obs_grp.create_dataset(name, data=arr, dtype='uint8', chunks=chunk_size)
                        else:
                            obs_grp.create_dataset(name, data=arr, dtype='uint8')

                    else:
                        obs_grp.create_dataset(name, data=np.array(v))

                except Exception as e:
                    print(f"{RED}Error saving observation dataset {k}: {e}{RESET}")

            for k, v in data_dict.items():
                if k.startswith("/action/"):
                    try:
                        act_grp.create_dataset(k.replace("/action/", ""), data=np.array(v))
                    except Exception as e:
                        print(f"{RED}Error saving action dataset {k}: {e}{RESET}")

            for k, v in data_dict.items():
                if not k.startswith("/labels/"):
                    continue
                name = k.replace("/labels/", "")
                if name == "rewards":
                    continue
                try:
                    if name == "task_done":
                        lbl_grp.create_dataset(name, data=np.array(v, dtype=bool))
                    elif name == "status":
                        str_array = np.array(v, dtype='S')
                        lbl_grp.create_dataset(name, data=str_array)
                    else:
                        lbl_grp.create_dataset(name, data=np.array(v))
                except Exception as e:
                    print(f"{RED}Error saving label dataset {k}: {e}{RESET}")

            T = meta.get('/metadata/total_frames', last_step)
            if "/rewards" in data_dict:
                rewards = np.array(data_dict["/rewards"], dtype=np.float32)
            else:
                rewards = np.zeros(T, dtype=np.float32)

            if rewards.size == 0:
                rewards = np.zeros(T, dtype=np.float32)
            elif rewards.shape[0] < T:
                rewards = np.pad(rewards, (0, T - rewards.shape[0]), mode="constant")
            elif rewards.shape[0] > T:
                rewards = rewards[:T]

            try:
                lbl_grp.create_dataset("rewards", data=rewards)
            except Exception as e:
                print(f"{RED}Error saving reward dataset: {e}{RESET}")

            try:
                metadata_mapping = [
                    ("/metadata/operator",              "operator",             ""),
                    ("/metadata/date",                  "collected_at",         ""),
                    ("/metadata/edited_date",           "edited_at",            ""),
                    ("/metadata/data_collection_error", "data_collection_error", False),
                    ("/metadata/data_collection_method","data_collection_method",""),
                    ("/metadata/robot_type",            "robot_type",           ""),
                    ("/metadata/HZ",                    "HZ",                   0),
                    ("/metadata/total_frames",          "total_frames",         0),
                    ("/metadata/head_pos/pan",            "head_pos_pan",         0),
                    ("/metadata/head_pos/tilt",           "head_pos_tilt",        task.current_tilt),
                    ("/metadata/object_idx",            "object_idx",           0),
                    ("/metadata/object_num",            "object_num",           0),
                    ("/metadata/table_height",         "table_height",         0.0),
                    ("/metadata/using_hand",           "using_hand",           ""),
                    ("/metadata/body_angle",           "body_angle",           0.0)
                ]

                for old_key, new_key, default_val in metadata_mapping:
                    if old_key in meta:
                        v = meta[old_key]
                    else:
                        v = default_val
                    if isinstance(v, np.generic):
                        v = v.item()
                    meta_grp.create_dataset(new_key, data=v)

                if "/metadata/image_compressed" in meta:
                    raw = meta["/metadata/image_compressed"]
                else:
                    raw = meta.get("/metadata/compression", True)

                if isinstance(raw, (np.bool_, bool)):
                    img_comp = bool(raw)
                elif isinstance(raw, (bytes, np.bytes_)):
                    s = raw.decode("utf-8")
                    img_comp = s.strip().lower() in ("1", "true", "t", "yes", "y")
                elif isinstance(raw, str):
                    img_comp = raw.strip().lower() in ("1", "true", "t", "yes", "y")
                else:
                    img_comp = bool(raw)

                meta_grp.create_dataset("image_compressed", data=img_comp)

                if "/metadata/image_compression_ratio" in meta:
                    raw_ratio = meta["/metadata/image_compression_ratio"]
                else:
                    raw_ratio = meta.get("/metadata/compression", 0.0)

                try:
                    ratio = float(raw_ratio)
                except Exception:
                    ratio = 0.0

                meta_grp.create_dataset("image_compression_ratio", data=ratio)

                new_metadata_fields = {
                    "data_schema": {"datasets": [
                        {"path": "action/hand_joint_pos/left", "dimension": 6},
                        {"path": "action/hand_joint_pos/right", "dimension": 6},
                        {"path": "action/joint_pos/left", "dimension": 6},
                        {"path": "action/joint_pos/right", "dimension": 6},
                        {"path": "observation/hand_joint_cur/left", "dimension": 6},
                        {"path": "observation/hand_joint_cur/right", "dimension": 6},
                        {"path": "observation/hand_joint_pos/left", "dimension": 6},
                        {"path": "observation/hand_joint_pos/right", "dimension": 6},
                        {"path": "observation/joint_cur/left", "dimension": 6},
                        {"path": "observation/joint_cur/right", "dimension": 6},
                        {"path": "observation/joint_pos/left", "dimension": 6},
                        {"path": "observation/joint_pos/right", "dimension": 6},
                        {"path": "observation/quaternion/left", "dimension": 4},
                        {"path": "observation/quaternion/right", "dimension": 4},
                        {"path": "observation/xpos/left", "dimension": 3},
                        {"path": "observation/xpos/right", "dimension": 3}
                    ]},
                    "graph": {"version": 1, "series": [
                        {"path": "/action/hand_joint_pos/left", "category": "action", "name": "hand_joint_pos.left", "dtype": "float32"},
                        {"path": "/action/hand_joint_pos/right", "category": "action", "name": "hand_joint_pos.right", "dtype": "float32"},
                        {"path": "/action/joint_pos/left", "category": "action", "name": "joint_pos.left", "dtype": "float32"},
                        {"path": "/action/joint_pos/right", "category": "action", "name": "joint_pos.right", "dtype": "float32"},
                        {"path": "/observation/hand_joint_cur/left", "category": "observation", "name": "hand_joint_cur.left", "dtype": "float32"},
                        {"path": "/observation/hand_joint_cur/right", "category": "observation", "name": "hand_joint_cur.right", "dtype": "float32"},
                        {"path": "/observation/hand_joint_pos/left", "category": "observation", "name": "hand_joint_pos.left", "dtype": "float32"},
                        {"path": "/observation/hand_joint_pos/right", "category": "observation", "name": "hand_joint_pos.right", "dtype": "float32"},
                        {"path": "/observation/joint_cur/left", "category": "observation", "name": "joint_cur.left", "dtype": "float32"},
                        {"path": "/observation/joint_cur/right", "category": "observation", "name": "joint_cur.right", "dtype": "float32"},
                        {"path": "/observation/joint_pos/left", "category": "observation", "name": "joint_pos.left", "dtype": "float32"},
                        {"path": "/observation/joint_pos/right", "category": "observation", "name": "joint_pos.right", "dtype": "float32"},
                        {"path": "/observation/quaternion/left", "category": "observation", "name": "quaternion.left", "dtype": "float32"},
                        {"path": "/observation/quaternion/right", "category": "observation", "name": "quaternion.right", "dtype": "float32"},
                        {"path": "/observation/xpos/left", "category": "observation", "name": "xpos.left", "dtype": "float32"},
                        {"path": "/observation/xpos/right", "category": "observation", "name": "xpos.right", "dtype": "float32"}
                    ]},
                    "image_channels": {
                        "head":  {"path": "/observation/images/head"},
                        "left":  {"path": "/observation/images/left"},
                        "right": {"path": "/observation/images/right"}
                    },
                    "inspector": "",
                    "labels": {"version": 1, "series": [
                        {"id": "rewards",   "display_name": "Rewards",   "path": "/labels/rewards",   "dtype": "float32", "kind": "timeseries"},
                        {"id": "task_done", "display_name": "Task Done", "path": "/labels/task_done", "dtype": "bool",    "kind": "timeseries"}
                    ], "tasks": []},
                    "robot_3d_paths": {
                        "observation": {
                            "joint_pos": {
                                "left":  "observation/joint_pos/left",
                                "right": "observation/joint_pos/right"
                            },
                            "hand_pos": {
                                "left":  "observation/hand_joint_pos/left",
                                "right": "observation/hand_joint_pos/right"
                            }
                        },
                        "action": {
                            "joint_pos": {
                                "left":  "action/joint_pos/left",
                                "right": "action/joint_pos/right"
                            },
                            "hand_pos": {
                                "left":  "action/hand_joint_pos/left",
                                "right": "action/hand_joint_pos/right"
                            }
                        }
                    },
                }

                for key, value in new_metadata_fields.items():
                    if isinstance(value, dict):
                        json_str = json.dumps(value)
                        meta_grp.create_dataset(key, data=json_str.encode("utf-8"))
                    elif isinstance(value, str):
                        meta_grp.create_dataset(key, data=value.encode("utf-8"))
                    else:
                        meta_grp.create_dataset(key, data=value)

            except Exception as e:
                print(f"{YELLOW}Warning: Error saving metadata: {e}{RESET}")

            has_comp_lens = (
                isinstance(compressed_image_len, list) and len(compressed_image_len) > 0
            ) or (
                isinstance(compressed_image_len, np.ndarray) and compressed_image_len.size > 0
            )

            if config.get("compression", False) and has_comp_lens:
                f.create_dataset('compressed_image_len', data=compressed_image_len)

        for k in data_dict.keys():
            if isinstance(data_dict[k], list):
                data_dict[k].clear()
        data_dict.clear()
        gc.collect()

        elapsed_save = time.time() - start_save_time
        print(f"{GREEN}Episode {task.episode_index} saved at {dataset_path} ({elapsed_save:.1f}s){RESET}")

save_worker = None

class EpisodeControl:
    def __init__(self):
        self.task_state = "stop"
        self.skip = False
        self.lock = threading.Lock()

    def set_event(self, event: str):
        with self.lock:
            event = event.strip().strip("'\"")
            print(event.lower())
            if event == "#":
                print("debug: Received event '#' - Marking current episode as corrupted.")
                self.task_state = "mark"
            elif event == "$":
                time.sleep(0.3)

                if self.task_state == "start":
                    self.task_state = "stop"
                else:
                    self.task_state = "start"
            elif event == "^":
                    self.task_state = "stop"

            

def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    os.makedirs(dataset_dir, exist_ok=True)
    for i in range(10000):
        path = os.path.join(dataset_dir, f'{dataset_name_prefix}_episode_{i}.{data_suffix}')
        if not os.path.isfile(path):
            return i
    raise RuntimeError("Too many episodes (>10000)")

def spin_thread(executor: SingleThreadedExecutor):
    try:
        executor.spin()
    except Exception as e:
        print(f"Executor error: {e}")

_shutdown_requested = False

def signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print(f"\n{RED}Force exit!{RESET}")
        os._exit(1)
    _shutdown_requested = True
    print(f"\n{YELLOW}Received signal {signum}. Shutting down gracefully... (Ctrl+C again to force quit){RESET}")
    raise KeyboardInterrupt


def io_callback(msg, ctrl):
    try:
        ctrl.set_event(msg.data)
    except Exception as e:
        print(f"{RED}Error setting event: {e}{RESET}")


def record_episode(args, config, config_path, node, ctrl, stop_pub, ready_pub, cams, current_tilt_val):
    """Single episode recording function - returns immediately after queuing save"""
    global save_worker, engine

    head_cam, left_cam, right_cam = cams

    data_dict = {}
    for k in config["data_dict"].keys():
        data_dict[k] = []

    if '/labels/status' not in data_dict:
        data_dict['/labels/status'] = []

    steps = config.get("episode_len", 500)
    HZ = config.get("HZ", 30)
    DT = 1.0 / HZ

    # HZ 디버깅용 변수
    hz_debug = {
        "over_dt_count": 0,
        "max_elapsed": 0.0,
        "total_elapsed": 0.0,
        "elapsed_list": [],
        # 단계별 시간 측정
        "ros_get_dict": [],
        "ros_get_meta": [],
        "cam_capture": [],
        "img_resize": [],
        "img_encode": [],
        "data_append": [],
        "obs_check": [],
        "obs_append": [],
        "loop_overhead": [],  # tqdm + 루프 자체 오버헤드
        "sleep_time": [],     # 실제 sleep한 시간
        "memory_usage": [],   # (step, MB) 튜플 리스트
        # 연속 밀림 추적
        "consecutive_over": 0,
        "max_consecutive_over": 0,
        "over_dt_frames": []  # 밀린 프레임 번호 기록
    }

    image_compress = config.get("image_compressed", True)
    encode_param = None
    if image_compress:
        compression_value = 1.0
        jpeg_quality = int(100 * compression_value)
        encode_param = (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality)

    ctrl.task_state = "stop"
    print(f"{GREEN}Ready for recording. Waiting for START signal...{RESET}")

    while ctrl.task_state != "start":
        time.sleep(0.1)

    if ctrl.task_state == "start":
        
        print(f"{GREEN}Start recording for {steps} steps (HZ={HZ}) (DT={DT}){RESET}")

        empty_obs_count = 0
        max_empty_obs = 10 
    
        try:
            topics = node.get_topic_names_and_types()
            print(f"{YELLOW}Debug: Available topics: {topics}{RESET}")
            topic_names = [topic[0] for topic in topics] if topics else []
            print(f"{YELLOW}Debug: Topic names: {topic_names}{RESET}")
        except Exception as e:
            print(f"{YELLOW}Debug: Error getting topics: {e}{RESET}")
            topic_names = []
        
        stop_pub.publish(std_msgs.msg.Bool(data=False))
        ready_pub.publish(std_msgs.msg.Bool(data=False))


        print(f"{YELLOW}Aligning robot to master arm position...{RESET}")
        READY_EPS = 0.1
        READY_CYCLE = 10
        ready_count = 0

        while True:
            obs = node.get_dict()
            if not obs:
                time.sleep(0.02)
                continue

            action_left  = np.array(obs.get('/action/joint_pos/left',  []), dtype=float)
            action_right = np.array(obs.get('/action/joint_pos/right', []), dtype=float)
            obs_left     = np.array(obs.get('/observation/joint_pos/left',  []), dtype=float)
            obs_right    = np.array(obs.get('/observation/joint_pos/right', []), dtype=float)

            if action_left.size == 0 or obs_left.size == 0:
                time.sleep(0.02)
                continue

            # (A) 지금처럼 max abs
            # diff_left  = np.max(np.abs(action_left - obs_left))
            # diff_right = np.max(np.abs(action_right - obs_right))
            # ok = (diff_left < READY_EPS) and (diff_right < READY_EPS)

            # (B) C++처럼 L2 norm을 원하면
            q_cur = np.concatenate([obs_left, obs_right])
            q_tgt = np.concatenate([action_left, action_right])
            ok = (np.linalg.norm(q_cur - q_tgt) < READY_EPS)

            if ok:
                ready_count += 1
                if ready_count >= READY_CYCLE:
                    ready_pub.publish(std_msgs.msg.Bool(data=True))
                    print("Aligned -> published /align_ready=True")


                    break
            else:
                ready_count = 0

            # 중간에 stop/mark 들어오면 취소 + False publish
            if ctrl.task_state in ("stop", "mark"):
                ready_pub.publish(std_msgs.msg.Bool(data=False))
                return True

            time.sleep(0.05)
        print(f"{GREEN}Robot aligned to master arm position.{RESET}")
        # 비프음 + 녹화 시작
        play_beep("start")
        time.sleep(0.5)
        print(f"{GREEN}Collection Start !!{RESET}")
        last_step = steps

        # Image resize settings (루프 밖으로 이동)
        RESIZE_WIDTH = 640
        RESIZE_HEIGHT = 480
        ORIGINAL_SINGLE_WIDTH = 1600  # Single camera width before merge (1600x1200 → 3200x1200 merged)

        # 카메라별 캡처+리사이즈+인코딩 통합 함수 (루프 밖으로 이동)
        # head: 스테레오 (1280x480), left/right: 모노 (640x480)
        def process_camera(cam, cam_name):
            img = cam.get_image()
            if img is None:
                return None, cam_name

            if cam_name == "head":
                # 스테레오: split → resize 2번 → merge
                left_half = cv2.resize(img[:, :ORIGINAL_SINGLE_WIDTH], (RESIZE_WIDTH, RESIZE_HEIGHT))
                right_half = cv2.resize(img[:, ORIGINAL_SINGLE_WIDTH:], (RESIZE_WIDTH, RESIZE_HEIGHT))
                result = cv2.hconcat([left_half, right_half])
            else:
                # 모노 (left, right): 오른쪽 절반 사용 → resize
                result = cv2.resize(img[:, ORIGINAL_SINGLE_WIDTH:], (RESIZE_WIDTH, RESIZE_HEIGHT))

            # Encode
            if image_compress and encode_param is not None:
                _, enc = cv2.imencode(".jpg", result, encode_param)
                return enc.ravel(), cam_name
            return result, cam_name

        # ThreadPoolExecutor를 루프 밖에서 생성 (재사용)
        cam_executor = ThreadPoolExecutor(max_workers=3)

        try:
            t0, start_time = time.time(), time.time()
            #-------------------------------------Record Loop Start------------------------------
            for t in tqdm(range(steps)):
                _loop_start = time.time()

                # 1. ROS get_dict
                _t1 = time.time()
                obs = node.get_dict()
                hz_debug["ros_get_dict"].append(time.time() - _t1)

                # 2. ROS get_metadata
                _t1 = time.time()
                meta = node.get_metadata()
                hz_debug["ros_get_meta"].append(time.time() - _t1)

                try:
                    # 3+4+5. 병렬 처리 (캡처+리사이즈+인코딩)
                    _t1 = time.time()
                    futures = [
                        cam_executor.submit(process_camera, head_cam, "head"),
                        cam_executor.submit(process_camera, left_cam, "left"),
                        cam_executor.submit(process_camera, right_cam, "right"),
                    ]
                    results = {name: data for data, name in [f.result() for f in futures]}

                    head_enc = results["head"]
                    left_enc = results["left"]
                    right_enc = results["right"]
                    hz_debug["cam_capture"].append(time.time() - _t1)

                    if head_enc is None:
                        print(f"{RED}Warning: Head camera image is None at step {t}{RESET}")
                        ctrl.task_state = "mark"
                    if left_enc is None:
                        print(f"{RED}Warning: Left camera image is None at step {t}{RESET}")
                        ctrl.task_state = "mark"
                    if right_enc is None:
                        print(f"{RED}Warning: Right camera image is None at step {t}{RESET}")
                        ctrl.task_state = "mark"

                    # 6. Data append
                    _t1 = time.time()
                    data_dict[f'/observation/images/head'].append(head_enc)
                    data_dict[f'/observation/images/left'].append(left_enc)
                    data_dict[f'/observation/images/right'].append(right_enc)
                    data_dict[f'/labels/status'].append("pick_and_place")
                    hz_debug["data_append"].append(time.time() - _t1)

                except Exception as e:
                    print(f"{RED}Error capturing images from cameras: {e}{RESET}")
                    sys.exit(1)

                # 7. obs 체크 로직
                _t1 = time.time()
                if not obs:
                    empty_obs_count += 1
                    if empty_obs_count == 1:
                        print(f"{YELLOW}Debug: obs is empty. Keys: {list(obs.keys()) if obs else 'None'}{RESET}")
                        print(f"{YELLOW}Debug: meta: {meta}{RESET}")
                        print(f"{YELLOW}Debug: Checking individual topic data...{RESET}")
                        for key in data_dict.keys():
                            try:
                                topic_data = node.get_topic_data(key)
                                if topic_data is not None:
                                    print(f"{GREEN}Debug: Topic {key}: {type(topic_data)} - Data available{RESET}")
                                else:
                                    print(f"{YELLOW}Debug: Topic {key}: No data available{RESET}")
                            except Exception as e:
                                print(f"{RED}Debug: Error getting topic {key}: {e}{RESET}")

                    if empty_obs_count >= max_empty_obs:
                        print(f"{RED}Warning: No data received for {max_empty_obs} consecutive steps. Stopping collection.{RESET}")
                        print(f"{RED}Debug: Last obs: {obs}{RESET}")
                        print(f"{RED}Debug: Last meta: {meta}{RESET}")
                        stop_pub.publish(std_msgs.msg.Bool(data=True))
                        nodata = True
                        break
                else:
                    empty_obs_count = 0
                    if t == 0:
                        print(f"{GREEN}Debug: First obs received. Keys: {list(obs.keys())}{RESET}")
                        print(f"{GREEN}Debug: First obs sample: {dict(list(obs.items())[:3])}{RESET}")
                        pass
                hz_debug["obs_check"].append(time.time() - _t1)

                # 8. obs append 루프
                _t1 = time.time()
                for k in data_dict.keys():
                    if k in obs:
                        data_dict[k].append(obs[k])
                hz_debug["obs_append"].append(time.time() - _t1)

                # 루프 내 실제 작업시간 (loop_start ~ 지금)
                _loop_work_time = time.time() - _loop_start

                t1 = time.time()
                elapsed = t1 - t0

                # 측정된 작업시간 합계
                measured_total = (
                    hz_debug["ros_get_dict"][-1] +
                    hz_debug["ros_get_meta"][-1] +
                    hz_debug["cam_capture"][-1] +
                    hz_debug["data_append"][-1] +
                    hz_debug["obs_check"][-1] +
                    hz_debug["obs_append"][-1]
                )

                # loop_work_time vs measured_total 차이 = 측정 사이 갭
                intra_loop_gap = _loop_work_time - measured_total

                # elapsed vs loop_work_time 차이 = 프레임 간 갭 (이전 sleep 후 ~ 현재 루프 시작)
                inter_frame_gap = elapsed - _loop_work_time

                hz_debug["loop_overhead"].append(inter_frame_gap)  # 프레임 간 갭

                # 새로운 디버그: intra_loop_gap 저장
                if "intra_loop_gap" not in hz_debug:
                    hz_debug["intra_loop_gap"] = []
                hz_debug["intra_loop_gap"].append(intra_loop_gap)

                # HZ 디버깅 로그 수집
                hz_debug["total_elapsed"] += elapsed
                hz_debug["elapsed_list"].append(elapsed)
                if elapsed > hz_debug["max_elapsed"]:
                    hz_debug["max_elapsed"] = elapsed

                if elapsed < DT:
                    sleep_start = time.time()
                    time.sleep(DT - elapsed)
                    hz_debug["sleep_time"].append(time.time() - sleep_start)
                    # 연속 밀림 리셋
                    hz_debug["consecutive_over"] = 0
                else:
                    hz_debug["over_dt_count"] += 1
                    hz_debug["sleep_time"].append(0)
                    # 연속 밀림 추적
                    hz_debug["consecutive_over"] += 1
                    hz_debug["over_dt_frames"].append(t)
                    if hz_debug["consecutive_over"] > hz_debug["max_consecutive_over"]:
                        hz_debug["max_consecutive_over"] = hz_debug["consecutive_over"]
                    if hz_debug["over_dt_count"] <= 10:  # 처음 10개만 상세 출력
                        print(f"{YELLOW}Warning: Frame {t} exceeded DT. DT={DT:.4f}s, elapsed={elapsed:.4f}s (+{(elapsed-DT)*1000:.1f}ms) [연속 {hz_debug['consecutive_over']}]{RESET}")
                t0 = time.time()  # sleep 후 시간으로 갱신

                # 9. 메모리 추적 (50프레임마다, GC 제거됨)
                if t % 50 == 0 and t > 0:
                    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    hz_debug["memory_usage"].append((t, mem_mb))
                
                if ctrl.task_state == "mark":
                    run_camera = False
                    last_step = t+1
                    play_beep("stop")
                    print(f"{RED}Task finished at step {last_step}{RESET}")
                    break
                elif ctrl.task_state == "stop":
                    run_camera = False
                    last_step = t+1
                    play_beep("stop")
                    print(f"{YELLOW}Task stopped at step {last_step}{RESET}")
                    break
            #-------------------------------------Record Loop End------------------------------
            # DO NOT stop cameras - keep them running for next episode

            stop_pub.publish(std_msgs.msg.Bool(data=True))

            elapsed = time.time() - start_time
            collected_steps = len(data_dict["/labels/task_done"]) if "/labels/task_done" in data_dict else last_step
            freq = collected_steps / elapsed if elapsed > 0 else 0
            print(f"Collected {collected_steps} steps, {freq:.2f}HZ")

            # HZ 디버깅 결과 출력
            print(f"\n{YELLOW}====== HZ Debug Summary ======{RESET}")
            print(f"Target HZ: {HZ}, Target DT: {DT*1000:.2f}ms")
            print(f"Actual HZ: {freq:.2f}")
            if collected_steps > 0:
                print(f"Over DT count: {hz_debug['over_dt_count']} / {collected_steps} ({hz_debug['over_dt_count']/collected_steps*100:.1f}%)")
            else:
                print(f"Over DT count: {hz_debug['over_dt_count']} / 0 (no data collected)")
            print(f"Max consecutive over: {hz_debug['max_consecutive_over']}")
            if hz_debug["over_dt_frames"]:
                print(f"Over DT frames: {hz_debug['over_dt_frames'][:20]}{'...' if len(hz_debug['over_dt_frames']) > 20 else ''}")
            print(f"Max elapsed: {hz_debug['max_elapsed']*1000:.2f}ms (target: {DT*1000:.2f}ms)")
            if hz_debug["elapsed_list"]:
                avg_elapsed = hz_debug["total_elapsed"] / len(hz_debug["elapsed_list"])
                print(f"Avg elapsed: {avg_elapsed*1000:.2f}ms")
                sorted_elapsed = sorted(hz_debug["elapsed_list"])
                p50 = sorted_elapsed[len(sorted_elapsed)//2]
                p95 = sorted_elapsed[int(len(sorted_elapsed)*0.95)]
                p99 = sorted_elapsed[int(len(sorted_elapsed)*0.99)]
                print(f"Percentiles - p50: {p50*1000:.2f}ms, p95: {p95*1000:.2f}ms, p99: {p99*1000:.2f}ms")

            # 단계별 시간 분석
            print(f"\n{YELLOW}------ Step-by-step Breakdown ------{RESET}")
            steps_breakdown = [
                ("ROS get_dict", hz_debug["ros_get_dict"]),
                ("ROS get_meta", hz_debug["ros_get_meta"]),
                ("Cam capture", hz_debug["cam_capture"]),
                ("Data append", hz_debug["data_append"]),
                ("Obs check", hz_debug["obs_check"]),
                ("Obs append", hz_debug["obs_append"]),
                ("Intra-loop gap", hz_debug.get("intra_loop_gap", [])),  # 측정 사이 갭
                ("Inter-frame gap", hz_debug["loop_overhead"]),  # 프레임 간 갭
            ]
            total_avg = 0
            for name, times in steps_breakdown:
                if times:
                    avg = sum(times) / len(times) * 1000
                    max_t = max(times) * 1000
                    total_avg += avg
                    print(f"  {name:15s}: avg={avg:6.2f}ms, max={max_t:6.2f}ms")
            print(f"  {'TOTAL':15s}: avg={total_avg:6.2f}ms")

            # Sleep 시간 분석
            if hz_debug["sleep_time"]:
                avg_sleep = sum(hz_debug["sleep_time"]) / len(hz_debug["sleep_time"]) * 1000
                print(f"\n  Avg sleep time: {avg_sleep:.2f}ms (target DT - work time)")

            # 메모리 사용량 출력
            if hz_debug["memory_usage"]:
                print(f"\n{YELLOW}------ Memory Usage ------{RESET}")
                mem_start = hz_debug["memory_usage"][0][1] if hz_debug["memory_usage"] else 0
                mem_end = hz_debug["memory_usage"][-1][1] if hz_debug["memory_usage"] else 0
                mem_diff = mem_end - mem_start
                print(f"  Start: {mem_start:.1f} MB")
                print(f"  End:   {mem_end:.1f} MB")
                print(f"  Diff:  {mem_diff:+.1f} MB")
                if len(hz_debug["memory_usage"]) > 1:
                    steps_diff = hz_debug["memory_usage"][-1][0] - hz_debug["memory_usage"][0][0]
                    if steps_diff > 0:
                        mb_per_100_steps = (mem_diff / steps_diff) * 100
                        print(f"  Rate:  {mb_per_100_steps:+.2f} MB / 100 steps")

            # 카메라 드랍 통계 출력
            print(f"\n{YELLOW}------ Camera Drop Stats ------{RESET}")
            for cam, cam_name in [(head_cam, "head"), (left_cam, "left"), (right_cam, "right")]:
                stats = cam.get_stats()
                total = stats["total_reads"]
                drop1 = stats["drop_count1"]
                drop2 = stats["drop_count2"]
                max_cons1 = stats["max_consecutive1"]
                max_cons2 = stats["max_consecutive2"]
                drop_rate1 = (drop1 / total * 100) if total > 0 else 0
                drop_rate2 = (drop2 / total * 100) if total > 0 else 0
                print(f"  {cam_name:5s}: cam1 drop={drop1:4d} ({drop_rate1:5.2f}%), max_consecutive={max_cons1:3d}")
                print(f"  {cam_name:5s}: cam2 drop={drop2:4d} ({drop_rate2:5.2f}%), max_consecutive={max_cons2:3d}")
                cam.reset_stats()  # 다음 에피소드를 위해 리셋

            print(f"{YELLOW}=============================={RESET}\n")

            total_data_points = sum(len(v) for v in data_dict.values() if isinstance(v, list))
            if total_data_points == 0:
                print(f"{RED}Warning: data_dict is empty! No data was collected.{RESET}")
                return True  # Return True to continue loop

            print(f"{GREEN}Data collection successful. Total data points: {total_data_points}{RESET}")

            # Prepare metadata
            meta = meta if 'meta' in locals() and isinstance(meta, dict) else {}
            meta['/metadata/total_frames'] = last_step
            meta['/metadata/head_pos/pan'] = 0
            meta['/metadata/head_pos/tilt'] = current_tilt_val
            meta['/metadata/object_idx'] = node.object_idx
            meta['/metadata/object_num'] = node.object_num
            meta['/metadata/table_height'] = node.table_height
            meta['/metadata/using_hand'] = node.using_hand
            meta['/metadata/body_angle'] = node.body_angle
            meta['/metadata/labels'] = labels_string

            episode_index = config.get("episode_index", 1)
            task_state_snapshot = ctrl.task_state

            # Shallow copy: transfer ownership to save worker
            data_dict_for_save = {}
            for k, v in data_dict.items():
                data_dict_for_save[k] = v

            # Create save task
            save_task = SaveTask(
                data_dict=data_dict_for_save,
                meta=dict(meta),  # Copy meta
                config=dict(config),  # Copy config
                config_path=config_path,
                last_step=last_step,
                task_state=task_state_snapshot,
                current_tilt=current_tilt_val,
                episode_index=episode_index,
                task_name=args.task_name,
                image_compress=image_compress,
                encode_param=encode_param
            )

            # Wait for save slot if queue is full
            save_worker.wait_for_slot()

            # Submit to save worker (non-blocking)
            save_worker.submit(save_task)

            # Update config for next episode immediately and save to file
            config["episode_index"] = episode_index + 1
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"{YELLOW}Warning: Failed to save config: {e}{RESET}")

            print(f"{GREEN}Episode {episode_index} queued for saving. Ready for next recording!{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Error during episode recording: {e}{RESET}")
            import traceback
            print(f"{RED}Error details: {traceback.format_exc()}{RESET}")
            return True  # Continue to next episode

        finally:
            # ThreadPoolExecutor 정리
            if 'cam_executor' in locals():
                cam_executor.shutdown(wait=False)

    return True


def main(args):
    """Main function - initializes resources once and runs episode loop"""
    global save_worker, engine

    config_path = os.path.join(os.path.dirname(__file__), "config_test", args.task_name + ".json")
    config = load_config(config_path)

    robot_id = config["robot_id"]

    # Initialize cameras ONCE (keep running throughout)
    print(f"{GREEN}Initializing cameras...{RESET}")
    head_cam = RBRSCamera(device_id1='/dev/head_camera1', device_id2='/dev/head_camera2')
    left_cam = RBRSCamera(device_id1='/dev/left_camera1', device_id2='/dev/left_camera2')
    right_cam = RBRSCamera(device_id1='/dev/right_camera1', device_id2='/dev/right_camera2')
    cams = [head_cam, left_cam, right_cam]

    for cam in cams:
        try:
            cam.start()
        except Exception as e:
            print(f"{RED}Error starting camera: {e}{RESET}")
            return False

    # Initialize ROS2 components ONCE
    try:
        node = GenericRecorder(config)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        ctrl = EpisodeControl()
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        ready_pub = node.create_publisher(
            std_msgs.msg.Bool, f"/igris_b/{robot_id}/ready", 10
        )
        
        node.io_subscription = node.create_subscription(
            std_msgs.msg.String,
            f"/igris_b/{robot_id}/io_event",
            lambda msg: io_callback(msg, ctrl),
            qos
        )
        stop_pub = node.create_publisher(std_msgs.msg.Bool, f"/igris_b/{robot_id}/stop", 10)
        

    except Exception as e:
        print(f"{RED}Error initializing ROS2 components: {e}{RESET}")
        for cam in cams:
            cam.stop()
        return False

    # Start executor thread
    ros_thread = threading.Thread(target=spin_thread, args=(executor,))
    ros_thread.daemon = True
    ros_thread.start()
    time.sleep(1.0)

    # Initialize save worker
    save_worker = SaveWorker()

    # Initialize TTS engine
    engine = pyttsx3.init()
    engine.say("IGRIS READY")
    engine.runAndWait()

    # Preload beep sound (blocking to ensure audio system is ready)
    try:
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0)
    except:
        pass

    print(f"{GREEN}Data Collection Initialized{RESET}")
    print(f"{YELLOW}Waiting for io_event: '$' to START, '^' to STOP, '#' to MARK{RESET}")

    # Episode loop
    episode_count = 0
    tilt_list = [150, 150, 150]
    tilt_idx = 0

    try:
        while True:
            episode_count += 1
            print(f"\n{GREEN}=== Episode {episode_count} ==={RESET}")

            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            print(f"{YELLOW}Memory: {memory_before:.1f} MB (Save queue: {save_worker.get_pending_count()}){RESET}")

            set_head_pan(180)
            current_tilt = tilt_list[tilt_idx]
            set_head_tilt(current_tilt)

            # Reload config to get updated episode_index
            config = load_config(config_path)

            success = record_episode(
                args=args,
                config=config,
                config_path=config_path,
                node=node,
                ctrl=ctrl,
                stop_pub=stop_pub,
                ready_pub=ready_pub,
                cams=cams,
                current_tilt_val=current_tilt
            )

            if not success:
                print(f"{RED}Episode failed, continuing...{RESET}")

            gc.collect()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user{RESET}")

    finally:
        print(f"{YELLOW}Cleaning up...{RESET}")

        # Wait for pending saves to complete
        if save_worker:
            pending = save_worker.get_pending_count()
            if pending > 0:
                print(f"{YELLOW}Waiting for {pending} pending saves to complete...{RESET}")
            save_worker.shutdown()

        # Stop cameras
        for cam in cams:
            try:
                cam.stop()
            except:
                pass

        cv2.destroyAllWindows()

        # Shutdown ROS2
        try:
            executor.shutdown(timeout_sec=2.0)
            node.destroy_node()
        except:
            pass

        set_head_torque(False)
        print(f"{GREEN}Cleanup complete{RESET}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--task_name", type=str, required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rclpy.init()

    try:
        portHandler.openPort()
    except Exception as e:
        print(f"{RED}Error opening port: {e}{RESET}")
        sys.exit(1)
    BAUDRATE = 1000000
    try:
        portHandler.setBaudRate(BAUDRATE)
    except Exception as e:
        print(f"{RED}Error setting baudrate: {e}{RESET}")
        sys.exit(1)
    set_head_torque(True)

    try:
        main(args)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Program interrupted by user{RESET}")
        set_head_torque(False)
    except Exception as e:
        print(f"{RED}Unexpected error: {e}{RESET}")
        set_head_torque(False)
    finally:
        print(f"{GREEN}Program terminated{RESET}")
        set_head_torque(False)
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"{YELLOW}Warning: Error during ROS2 shutdown: {e}{RESET}")