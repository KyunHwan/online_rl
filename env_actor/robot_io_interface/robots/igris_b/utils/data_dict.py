#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import importlib
import time

RESET = "\033[0m"
RED = "\033[31m"

def get_msg_class(msg_type: str):
    mapping = {
        "PoseStamped": "geometry_msgs.msg",
        "JointState": "sensor_msgs.msg",
        "Float32MultiArray": "std_msgs.msg",
        "Image": "sensor_msgs.msg",
        "Bool": "std_msgs.msg",
        "Int32": "std_msgs.msg",
        "Float32": "std_msgs.msg",
        "Int64": "std_msgs.msg",
        "Float64": "std_msgs.msg",
        "String": "std_msgs.msg",
    }
    for k, v in mapping.items():
        if msg_type == k:
            module = importlib.import_module(v)
            return getattr(module, msg_type)
    raise ValueError(f"Unknown msg_type {msg_type}")

class GenericRecorder(Node):
    def __init__(self, config):
        super().__init__('dict_recorder')
        self.config = config
        self.data = {}
        self.data_time = {}
        self.data_time_last = time.time()
        self.robot_id = config["robot_id"]
        self.HZ = config["HZ"]
        self.DT = 1.0 / self.HZ
        
        print(f"robot_id: {self.robot_id}")
        print(f"HZ: {self.HZ}")
        
        self.timer = self.create_timer(self.DT * 10, self.timer_cb)

        # print("=== Subscribing Topics & Fields ===")
        for name, cfg in config["topics"].items():
            msg_cls = get_msg_class(cfg["msg_type"])
            
            # Initialize data_time and data dictionaries for each field
            for key in cfg["fields"].keys():
                self.data_time[key] = []
                self.data[key] = None
            
            self.create_subscription(
                msg_cls,
                cfg["topic"],
                lambda msg, n=name, c=cfg: self.cb(n, msg, c),
                10
            )
        
        print(f"Total subscriptions created: {len(config['topics'])}")
        
    def timer_cb(self):
        # Check if the image data is all subscribed
        for key, cfg in self.config["topics"].items():
            if cfg["msg_type"] == "Image":
                # Check each field for this topic
                for field_key in cfg["fields"].keys():
                    if self.data[field_key] is None:
                        print(f"{RED}Warning: Image data {field_key} is not subscribed{RESET}")
                    else:
                        if self.data_time[field_key][-1] > self.DT * 10:
                            print(f"{RED}Warning: Image data {field_key} time is greater than DT. DT={self.DT}, data_time={self.data_time[field_key][-1]}{RESET}")

    def cb(self, topic_name, msg, cfg):
        for key, rule in cfg["fields"].items():
            if isinstance(rule, str):
                parts = rule.split(".")
                val = msg
                for p in parts:
                    val = getattr(val, p)

                if "position" in rule:
                    arr = np.array([val.x, val.y, val.z], dtype=np.float32)
                elif "orientation" in rule:
                    arr = np.array([val.x, val.y, val.z, val.w], dtype=np.float32)
                else:
                    if hasattr(val, 'typecode') and hasattr(val, 'tobytes'):
                        arr = np.array(val, dtype=np.uint8)
                        if hasattr(msg, 'height') and hasattr(msg, 'width'):
                            try:
                                if hasattr(msg, 'encoding') and msg.encoding == 'rgb8':
                                    arr = arr.reshape(msg.height, msg.width, 3)
                                else:
                                    arr = arr.reshape(msg.height, msg.width)
                            except Exception as e:
                                print(f"Warning: Failed to reshape image data: {e}")
                    else:
                        arr = val
                
                self.data[key] = arr
                self.data_time[key].append(time.time() - self.data_time_last)

            elif isinstance(rule, dict):
                attr = rule.get("attr", "data")
                arr = getattr(msg, attr)
                
                if "slice" in rule:
                    s, e = rule["slice"]
                    arr = np.array(arr[s:e], dtype=np.float32)
                
                self.data[key] = arr
                self.data_time[key].append(time.time() - self.data_time_last)
            
            # Check timing
            if self.data_time[key][-1] > self.DT:
                print(f"{RED}Warning: Data {key} time is greater than DT. DT={self.DT}, data_time={self.data_time[key][-1]}{RESET}")
        
        self.data_time_last = time.time()

    def get_dict(self):
        return self.data
    
    def get_observation_dict(self):
        return {k: v for k, v in self.data.items() if k.startswith("/observation/")}
    
