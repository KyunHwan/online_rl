#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header, Float32MultiArray
import socket
import threading
import datetime
import numpy as np
import json
import os
from ament_index_python.packages import get_package_prefix

JOINT_NAMES = ['r_hand_base', 'r_joint_thumb_CMC', 'r_joint_thumb_PP', 'r_joint_thumb_DP', 'r_joint_index_PP', 'r_joint_index_DP', 'r_joint_middle_PP', 'r_joint_middle_DP', 'r_joint_ring_PP', 'r_joint_ring_DP', 'r_joint_little_PP', 'r_joint_little_DP',
               'l_hand_base', 'l_joint_thumb_CMC', 'l_joint_thumb_PP', 'l_joint_thumb_DP', 'l_joint_index_PP', 'l_joint_index_DP', 'l_joint_middle_PP', 'l_joint_middle_DP', 'l_joint_ring_PP', 'l_joint_ring_DP', 'l_joint_little_PP', 'l_joint_little_DP']

THUMB_INDICES = {'CMC': 1, 'PP': 2, 'DP': 3}  
# data format
# JOINT1/JOINT2/JOINT3/SPREAD
# 0~3 : Thumb
# 4~7 : Index
# 8~11 : Middle
# 12~15 : Ring
# 16~19 : Little

def deg2rad(deg):
    return deg * np.pi / 180.0

def load_calibration_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Calibration file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def normalize_value(value, min_val, max_val, target_min=0.0, target_max=1.0):
    """
    normalize a value from [min_val, max_val] to [target_min, target_max].
    """
    if max_val - min_val == 0:
        return target_min
    value = np.clip(value, min_val, max_val)  # Ensure value is within bounds
    normalized_val = (value - min_val) / (max_val - min_val)
    return normalized_val

def apply_hold_zone(value, hold_value, tolerance=0.2):
    """
    value가 hold_value ± tolerance 안에 있으면 hold_value로 고정
    """
    if abs(value - hold_value) <= tolerance:
        return hold_value
    return value

def invert_normalize_value(value, min_val, max_val, target_min=0.0, target_max=1.0):
    """
    invert normalize a value from [min_val, max_val] to [target_min, target_max].
    """
    if max_val - min_val == 0:
        return target_min
    value = np.clip(value, max_val, min_val)  # Ensure value is within bounds
    value = np.interp(value, [max_val, min_val], [min_val, max_val])
    normalized_val = (value - max_val) / (min_val - max_val)
    return normalized_val

class ManusUDPReceiver(Node):
    def __init__(self):
        super().__init__('manus_udp_receiver')
        
        # Declare parameters
        self.declare_parameter('left_glove_port', 19901)
        self.declare_parameter('right_glove_port', 19902)
        self.declare_parameter('publish_rate', 100.0)

        # Operator name (REQUIRED)
        self.declare_parameter('operator_name', '')
        operator_name = self.get_parameter('operator_name').value

        if operator_name == '':
            self.get_logger().fatal(
                'operator_name parameter is REQUIRED.\n'
                'Example:\n'
                '  ros2 run manus2ros manus_udp_receiver '
                '--ros-args -p operator_name:=KIM'
            )
            rclpy.shutdown()
            return

        # Config path
        workspace_dir = os.path.expanduser("~/colcon_ws")
        config_dir = os.path.join(
            workspace_dir,
            "src", "IGRIS_B_CASHIER", "ros2", "manus2ros", "config"
        )
        os.makedirs(config_dir, exist_ok=True)

        # Calibration file path
        filename = f"calibration_{operator_name}.json"
        self.calibration_file = os.path.join(config_dir, filename)

        # Create calibration file if missing
        if not os.path.exists(self.calibration_file):
            default_data = {}
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
            self.get_logger().info(
                f"Calibration file not found, created new: {self.calibration_file}"
            )
        else:
            self.get_logger().info(
                f"Using existing calibration file: {self.calibration_file}"
            )

        # Calibration default min/max values
        self.calib_hands_down_values = {'left': [0.0]*20, 'right': [0.0]*20}
        self.calib_thumbCMC_down_values = {'left': [0.0]*20, 'right': [0.0]*20}
        self.calib_grap_values = {'left': [70.0]*20, 'right': [70.0]*20}

        # Load calibration file
        try:
            calib_data = load_calibration_json(self.calibration_file)
            self.calib_hands_down_values['left'] = calib_data['hand_down']['left']
            self.calib_thumbCMC_down_values['left'] = calib_data['thumbCMC_down']['left']
            self.calib_grap_values['left'] = calib_data['grap']['left']
            self.calib_hands_down_values['right'] = calib_data['hand_down']['right']
            self.calib_thumbCMC_down_values['right'] = calib_data['thumbCMC_down']['right']
            self.calib_grap_values['right'] = calib_data['grap']['right']
            self.get_logger().info("Calibration data loaded successfully.")
        except Exception as e:
            self.get_logger().error(
                f"Calibration load error: {e}. Using default ranges."
            )

        # Read parameters
        self.left_port = self.get_parameter('left_glove_port').value
        self.right_port = self.get_parameter('right_glove_port').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Publishers
        self.left_pose_pub = self.create_publisher(
            PoseArray, 'left_glove/finger_poses_raw', 10
        )
        self.right_pose_pub = self.create_publisher(
            PoseArray, 'right_glove/finger_poses_raw', 10
        )
        self.target_finger_pub = self.create_publisher(
            Float32MultiArray, '/finger_target', 10
        )
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', 10
        )
        
        # Data storage
        self.left_data = None
        self.right_data = None
        self.left_lock = threading.Lock()
        self.right_lock = threading.Lock()

        # Timer
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_data
        )

        # Start UDP listeners
        self.start_udp_listeners()

        self.get_logger().info(
            f'Manus UDP Receiver started '
            f'(operator: {operator_name}, '
            f'left port: {self.left_port}, '
            f'right port: {self.right_port})'
        )

    def start_udp_listeners(self):
        """Start UDP listener threads for both gloves"""
        # Left glove listener
        self.left_thread = threading.Thread(target=self.udp_listener, args=(self.left_port, 'left'))
        self.left_thread.daemon = True
        self.left_thread.start()
        
        # Right glove listener
        self.right_thread = threading.Thread(target=self.udp_listener, args=(self.right_port, 'right'))
        self.right_thread.daemon = True
        self.right_thread.start()
    
    def udp_listener(self, port, glove_side):
        """UDP listener thread for receiving glove data"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(1.0)
        
        self.get_logger().info(f'Started UDP listener for {glove_side} glove on port {port}')
        
        while rclpy.ok():
            try:
                data, addr = sock.recvfrom(1024)
                text = data.decode().strip()
                values = list(map(float, text.split()))
                
                if len(values) >= 20:  # 5 fingers * 4 values each
                    if glove_side == 'left':
                        with self.left_lock:
                            self.left_data = values
                    else:
                        with self.right_lock:
                            self.right_data = values
                                    
                    self.get_logger().debug(f'Received {glove_side} glove data: {len(values)} values')
                    
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'Error receiving {glove_side} glove data: {str(e)}')
                continue
    
    def publish_data(self):
        """Publish glove data as ROS2 messages"""
        current_time = self.get_clock().now()
        
        # Publish left glove data
        if self.left_data is not None:
            with self.left_lock:
                left_values = self.left_data.copy()
            
            # Publish finger poses
            left_pose_msg = self.create_pose_array_msg('left_glove', left_values, current_time)
            self.left_pose_pub.publish(left_pose_msg)
        
        # Publish right glove data
        if self.right_data is not None:
            with self.right_lock:
                right_values = self.right_data.copy()
            
            # Publish finger poses
            right_pose_msg = self.create_pose_array_msg('right_glove', right_values, current_time)
            self.right_pose_pub.publish(right_pose_msg)
            
        if self.left_data and self.right_data:
            with self.left_lock:
                left_values = self.left_data.copy()
            with self.right_lock:
                right_values = self.right_data.copy()
            target_finger_msg = self.create_target_finger_msg(left_values, right_values)
            self.target_finger_pub.publish(target_finger_msg)
            #joint_state_msg = self.create_target_finger_msg(left_values, right_values)
            joint_state_msg = self.create_joint_state_msg(left_values, right_values, current_time)
            # self.joint_state_pub.publish(joint_state_msg)
            
            self.get_logger().info(f'Left data [{len(left_values)}]')
            self.get_logger().info(f'Right data [{len(right_values)}]')
            
    def create_target_finger_msg(self, left_values, right_values):
        """Create a simplified Float32MultiArray message with a single value per finger."""
        msg = Float32MultiArray()

        # check if both gloves have sufficient data
        if len(left_values) < 20 or len(right_values) < 20:
            self.get_logger().warn("Insufficient glove data for target message.")
            return msg 

        # average finger values
        right_finger_data = []
        for i in range(5):
            base = i * 4
            if(i==0):
                
                joint_cmc = right_values[base + 1]
                joint_sum = (-right_values[base + 1] + right_values[base + 2] + right_values[base + 3])/3

                right_finger_data.append(joint_sum)
            else:
                joint_sum = (right_values[base + 2] + right_values[base + 3])/2
                right_finger_data.append(joint_sum)

        right_finger_data.append(joint_cmc) 

        left_finger_data = []
        for i in range(5):
            base = i * 4
            if(i==0):
                joint_cmc = left_values[base + 1]
                joint_sum = (-left_values[base + 1] + left_values[base + 2] + left_values[base + 3])/3

                left_finger_data.append(joint_sum)  
            else:
                # average 2-5 fingers 
                joint_sum = (left_values[base + 2] + left_values[base + 3])/2
                left_finger_data.append(joint_sum)

        left_finger_data.append(joint_cmc)
   
        # cali

        combined_data = []
        # right hand
        for i, val in enumerate(right_finger_data):
            # Thumb CMC is at index 0, others are at i*4+1
            if i == 5:
                min_val = self.calib_hands_down_values['right'][1]
                max_val = self.calib_thumbCMC_down_values['right'][1]
                normalized = invert_normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
            elif i == 0: # Thumb PP+DP
                min_val = (-self.calib_thumbCMC_down_values['right'][1] + self.calib_hands_down_values['right'][2] + self.calib_hands_down_values['right'][3])/3
                max_val = (-self.calib_grap_values['right'][1] + self.calib_grap_values['right'][2] + self.calib_grap_values['right'][3])/3
                normalized = normalize_value(val, min_val, max_val)
                # normalized = apply_hold_zone(normalized, 0.59, 0.2)
                combined_data.append(normalized)
            else: # Other fingers (index 1-4)
                # i-1 maps to finger 1 (index), 2 (middle), etc.
                # So we use (i-2)*4+1 for index, (i-2)*4+2 for middle, etc.
                finger_idx = i
                base_idx = finger_idx * 4
                min_val = (self.calib_thumbCMC_down_values['right'][base_idx + 2] + self.calib_thumbCMC_down_values['right'][base_idx + 3])/2
                max_val = (self.calib_grap_values['right'][base_idx + 2] + self.calib_grap_values['right'][base_idx + 3])/2
                normalized = normalize_value(val, min_val, max_val)
                # normalized = apply_hold_zone(normalized, 0.64, 0.2)
                combined_data.append(normalized)
        
        # left hand
        for i, val in enumerate(left_finger_data):
            if i == 5:
                min_val = self.calib_hands_down_values['left'][1]
                max_val = self.calib_thumbCMC_down_values['left'][1]
                normalized = invert_normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
            elif i == 0: # Thumb PP+DP
                min_val = (-self.calib_thumbCMC_down_values['left'][1] + self.calib_hands_down_values['left'][2] + self.calib_hands_down_values['left'][3])/3
                max_val = (-self.calib_grap_values['left'][1] + self.calib_grap_values['left'][2] + self.calib_grap_values['left'][3])/3
                normalized = normalize_value(val, min_val, max_val)
                # normalized = apply_hold_zone(normalized, 0.55,0.2)
                combined_data.append(normalized)
            else: # Other fingers (index 1-4)
                finger_idx = i
                base_idx = finger_idx * 4
                min_val = (self.calib_thumbCMC_down_values['left'][base_idx + 2] + self.calib_thumbCMC_down_values['left'][base_idx + 3])/2
                max_val = (self.calib_grap_values['left'][base_idx + 2] + self.calib_grap_values['left'][base_idx + 3])/2
                normalized = normalize_value(val, min_val, max_val)
                # normalized = apply_hold_zone(normalized, 0.64,0.2)
                combined_data.append(normalized)
                
        msg.data = combined_data
        return msg


    def create_joint_state_msg(self, left_values, right_values, timestamp):
        """
        Creates a JointState message with a single value per finger,
        correctly applying calibration.
        """
        msg = JointState()
        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = 'igris_b_handbase_link'
        msg.name = JOINT_NAMES
        
        right_finger_data = []
        for i in range(5):
            base = i * 4
            if(i==0):
                
                joint_cmc = right_values[base + 1]
                joint_sum = (-right_values[base + 1] + right_values[base + 2] + right_values[base + 3])/3

                right_finger_data.append(joint_sum)
            else:
                joint_sum = (right_values[base + 2] + right_values[base + 3])/2
                right_finger_data.append(joint_sum)
        
        right_finger_data.append(joint_cmc)

        left_finger_data = []
        for i in range(5):
            base = i * 4
            if(i==0):
                joint_cmc = left_values[base + 1]
                joint_sum = (-left_values[base + 1] + left_values[base + 2] + left_values[base + 3])/3

                left_finger_data.append(joint_sum)  
            else:
                # average 2-5 fingers 
                joint_sum = (left_values[base + 2] + left_values[base + 3])/2
                left_finger_data.append(joint_sum)
        left_finger_data.append(joint_cmc) 
           
        # cali
        combined_data = []
        # right hand
        for i, val in enumerate(right_finger_data):
            # Thumb CMC is at index 0, others are at i*4+1
            if i == 5:
                min_val = self.calib_hands_down_values['right'][1]
                max_val = self.calib_thumbCMC_down_values['right'][1]
                normalized = invert_normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
            elif i == 0: # Thumb PP+DP
                min_val = (-self.calib_hands_down_values['right'][1] + self.calib_hands_down_values['right'][2] + self.calib_hands_down_values['right'][3])/3
                max_val = (-self.calib_grap_values['right'][1] + self.calib_grap_values['right'][2] + self.calib_grap_values['right'][3])/3
                normalized = normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
            else: # Other fingers (index 1-4)
                # i-1 maps to finger 1 (index), 2 (middle), etc.
                # So we use (i-2)*4+1 for index, (i-2)*4+2 for middle, etc.
                finger_idx = i
                base_idx = finger_idx * 4
                min_val = (self.calib_thumbCMC_down_values['right'][base_idx + 2] + self.calib_thumbCMC_down_values['right'][base_idx + 3])/2
                max_val = (self.calib_grap_values['right'][base_idx + 2] + self.calib_grap_values['right'][base_idx + 3])/2
                normalized = normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
                
        # left hand
        for i, val in enumerate(left_finger_data):
            if i == 5:
                min_val = self.calib_hands_down_values['left'][1]
                max_val = self.calib_thumbCMC_down_values['left'][1]
                normalized = invert_normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
            elif i == 0: # Thumb PP+DP
                min_val = (-self.calib_hands_down_values['left'][1] + self.calib_hands_down_values['left'][2] + self.calib_hands_down_values['left'][3])/3
                max_val = (-self.calib_grap_values['left'][1] + self.calib_grap_values['left'][2] + self.calib_grap_values['left'][3])/3
                normalized = normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
            else: # Other fingers (index 1-4)
                finger_idx = i
                base_idx = finger_idx * 4
                min_val = (self.calib_thumbCMC_down_values['left'][base_idx + 2] + self.calib_thumbCMC_down_values['left'][base_idx + 3])/2
                max_val = (self.calib_grap_values['left'][base_idx + 2] + self.calib_grap_values['left'][base_idx + 3])/2
                normalized = normalize_value(val, min_val, max_val)
                combined_data.append(normalized)
        # Initialize a list of 24 joint positions with default values
        generalized_rad = [0.0] * 24

        # Map right hand joint data (indices 0-11)
        # We assume base joint has no rotational value
        generalized_rad[0] = 0.0
        generalized_rad[1] = deg2rad(combined_data[5]) # r_joint_thumb_CMC
        generalized_rad[2] = deg2rad(combined_data[0]) # r_joint_thumb_PP
        generalized_rad[3] = deg2rad(combined_data[0]) # r_joint_thumb_DP
        generalized_rad[4] = deg2rad(combined_data[1]) # r_joint_index_PP
        generalized_rad[5] = deg2rad(combined_data[1]) # r_joint_index_DP
        generalized_rad[6] = deg2rad(combined_data[2]) # r_joint_middle_PP
        generalized_rad[7] = deg2rad(combined_data[2]) # r_joint_middle_DP
        generalized_rad[8] = deg2rad(combined_data[3]) # r_joint_ring_PP
        generalized_rad[9] = deg2rad(combined_data[3]) # r_joint_ring_DP
        generalized_rad[10] = deg2rad(combined_data[4]) # r_joint_little_PP
        generalized_rad[11] = deg2rad(combined_data[4]) # r_joint_little_DP

        # Map left hand joint data (indices 12-23)
        # We assume base joint has no rotational value
        generalized_rad[12] = 0.0
        generalized_rad[13] = deg2rad(combined_data[11]) # l_joint_thumb_CMC
        generalized_rad[14] = deg2rad(combined_data[6]) # l_joint_thumb_PP
        generalized_rad[15] = deg2rad(combined_data[6]) # l_joint_thumb_DP
        generalized_rad[16] = deg2rad(combined_data[7]) # l_joint_index_PP
        generalized_rad[17] = deg2rad(combined_data[7]) # l_joint_index_DP
        generalized_rad[18] = deg2rad(combined_data[8]) # l_joint_middle_PP
        generalized_rad[19] = deg2rad(combined_data[8]) # l_joint_middle_DP
        generalized_rad[20] = deg2rad(combined_data[9]) # l_joint_ring_PP
        generalized_rad[21] = deg2rad(combined_data[9]) # l_joint_ring_DP
        generalized_rad[22] = deg2rad(combined_data[10]) # l_joint_little_PP
        generalized_rad[23] = deg2rad(combined_data[10]) # l_joint_little_DP

        # Final assignment to the message
        msg.position = generalized_rad

        return msg

        
    
    # def create_joint_state_msg(self, left_values, right_values, timestamp):
    #     """Create JointState message from glove data"""
    #     msg = JointState()
    #     msg.header.stamp = timestamp.to_msg()
    #     msg.header.frame_id = f'igris_b_handbase_link'
        
    #     right_rad = []
    #     for i in range(len(right_values)):
    #         right_rad.append(deg2rad(right_values[i]))
    #     left_rad = []
    #     for i in range(len(left_values)):
    #         left_rad.append(deg2rad(left_values[i]))
        
    #     msg.name = JOINT_NAMES
    #     # Initialize position array with zeros
    #     msg.position = [0.0] * len(JOINT_NAMES)
        
    #     # Set right hand joint positions
    #     msg.position[JOINT_NAMES.index('r_joint_thumb_CMC')] = (right_rad[0] + right_rad[1]) / 2
    #     msg.position[JOINT_NAMES.index('r_joint_thumb_PP')] = right_rad[2]
    #     msg.position[JOINT_NAMES.index('r_joint_thumb_DP')] = right_rad[3]
    #     msg.position[JOINT_NAMES.index('r_joint_index_PP')] = right_rad[5]
    #     msg.position[JOINT_NAMES.index('r_joint_index_DP')] = right_rad[6]
    #     msg.position[JOINT_NAMES.index('r_joint_middle_PP')] = right_rad[9]
    #     msg.position[JOINT_NAMES.index('r_joint_middle_DP')] = right_rad[10]
    #     msg.position[JOINT_NAMES.index('r_joint_ring_PP')] = right_rad[13]
    #     msg.position[JOINT_NAMES.index('r_joint_ring_DP')] = right_rad[14]
    #     msg.position[JOINT_NAMES.index('r_joint_little_PP')] = right_rad[17]
    #     msg.position[JOINT_NAMES.index('r_joint_little_DP')] = right_rad[18]

    #     # Set left hand joint positions
    #     msg.position[JOINT_NAMES.index('l_joint_thumb_CMC')] = (left_rad[0] + left_rad[1]) /2 
    #     msg.position[JOINT_NAMES.index('l_joint_thumb_PP')] = left_rad[2]
    #     msg.position[JOINT_NAMES.index('l_joint_thumb_DP')] = left_rad[3]
    #     msg.position[JOINT_NAMES.index('l_joint_index_PP')] = left_rad[5] 
    #     msg.position[JOINT_NAMES.index('l_joint_index_DP')] = left_rad[6] 
    #     msg.position[JOINT_NAMES.index('l_joint_middle_PP')] =left_rad[9]
    #     msg.position[JOINT_NAMES.index('l_joint_middle_DP')] = left_rad[10]
    #     msg.position[JOINT_NAMES.index('l_joint_ring_PP')] = left_rad[13] 
    #     msg.position[JOINT_NAMES.index('l_joint_ring_DP')] = left_rad[14] 
    #     msg.position[JOINT_NAMES.index('l_joint_little_PP')] = left_rad[17] 
    #     msg.position[JOINT_NAMES.index('l_joint_little_DP')] = left_rad[18]
        

    #     self.get_logger().info(f"r_joint_thumb_DP angle: {msg.position[JOINT_NAMES.index('r_joint_thumb_DP')]:.3f} rad")

    #     return msg
    
    def create_pose_array_msg(self, glove_side, values, timestamp):
        """Create PoseArray message from glove data"""
        msg = PoseArray()
        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = f'{glove_side}_base'
        
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, finger in enumerate(finger_names):
            pose = Pose()
            base = i * 4
            
            spread = values[base]
            joint1 = values[base + 1]
            joint2 = values[base + 2]
            joint3 = values[base + 3]
            
            pose.position.x = 0.1 * (i + 1)  
            pose.position.y = 0.05 * spread  
            pose.position.z = 0.02 * (joint1 + joint2 + joint3) 
            
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            
            msg.poses.append(pose)
        
        return msg


def main(args=None):
    rclpy.init(args=args)
    
    node = ManusUDPReceiver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 