from env_actor.runtime_settings_configs.igris_b.init_params import (
    INIT_JOINT_LIST,
    INIT_HAND_LIST,
    INIT_JOINT,
    IGRIS_B_STATE_KEYS
)
import numpy as np
import cv2
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState
from .utils.data_dict import GenericRecorder
from .utils.camera_utils import RBRSCamera
from env_actor.runtime_settings_configs.igris_b.inference_runtime_params import RuntimeParams

class ControllerBridge:
    """
    Stateless controller bridging between the robot and the controller interface class
    """
    def __init__(self,
                 runtime_params, 
                 inference_runtime_topics_config,):
        self.runtime_params = runtime_params

        if not rclpy.ok():
            rclpy.init()
        self.input_recorder = GenericRecorder(inference_runtime_topics_config)
        self.qos = QoSProfile(depth=10)
        self.qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.joint_pub = self.input_recorder.create_publisher(
                                JointState, 
                                f"/igris_b/{self.input_recorder.robot_id}/target_joints", 
                                qos_profile=self.qos
                        )
        self.finger_pub = self.input_recorder.create_publisher(
                                Float32MultiArray,
                                f"/igris_b/{self.input_recorder.robot_id}/finger_target", 
                                qos_profile=self.qos)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.input_recorder)
        self.cams = None
        self.recorder_thread = None

    @property
    def DT(self):
        return 1.0 / self.runtime_params.HZ
    
    @property
    def policy_update_period(self):
        return self.runtime_params.policy_update_period

    def recorder_rate_controller(self):
        return self.input_recorder.create_rate(self.runtime_params.HZ)

    def read_state(self) -> dict:
        """
        Returns proprio + images as numpy array
        """
        # proprio
        state_dict = {
            'proprio': self._obs_dict_to_np_array(self.input_recorder.get_observation_dict())
        }

        # images
        for cam_name in self.cams.keys():
            cam_image = self.cams[cam_name].get_image()
            if cam_image is not None:
                cam_image = cv2.resize(
                    cam_image,
                    dsize=(self.runtime_params.mono_img_resize_width, self.runtime_params.mono_img_resize_height),
                    interpolation=cv2.INTER_AREA,
                )
                state_dict[cam_name] = np.transpose(cam_image, (2, 0, 1)) # HWC -> CHW
            else: 
                print(f"{cam_name}: image is None !!")
        return state_dict
    
    def publish_action(self, action, prev_joint):
        # Receive prev_joint from Data Manager Bridge/Interface
        left_joint_pos   = action[:6]
        right_joint_pos  = action[6:12]
        left_finger_pos  = action[12:18]
        right_finger_pos = action[18:24]

        # Merge to robot joint order (environment-specific; verify mapping)
        raw_joint = np.concatenate([right_joint_pos, left_joint_pos])

        # Slew-rate limiting per joint (rad/step)
        delta = np.clip(raw_joint - prev_joint, -self.runtime_params.max_delta, self.runtime_params.max_delta)
        smoothed_joints = prev_joint + delta

        # Publish joints
        joint_msg = JointState()
        joint_msg.position = smoothed_joints.tolist()
        self.joint_pub.publish(joint_msg)

        # Publish fingers (right followed by left to match action layout)
        finger_msg = Float32MultiArray()
        finger_msg.data = list(right_finger_pos) + list(left_finger_pos)
        self.finger_pub.publish(finger_msg)

        # TODO: Implement returning smoothed_joints

        # This output should be used to update prev_joint in the Data Manager Bridge/Interface
        return smoothed_joints, np.concatenate([left_finger_pos, right_finger_pos])
    
    def start_state_readers(self):
        self._start_cam_recording()
        self._start_proprio_recording()
        self._check_proprio_reading()

    def init_robot_position(self):
        joint_msg = JointState()
        joint_msg.position = INIT_JOINT.copy()
        self.joint_pub.publish(joint_msg)
        
        return INIT_JOINT.copy()

    def _start_proprio_recording(self):
        self.recorder_thread = threading.Thread(target=self._spin_thread, args=(), daemon=True)
        self.recorder_thread.start()

    def _start_cam_recording(self):
        self.cams = {}
        for cam_name in self.runtime_params.camera_names:
            if cam_name in ['head', 'right']:
                self.cams[cam_name] = RBRSCamera(device_id1=f"/dev/{cam_name}_camera1", device_id2=None)
            elif cam_name == 'left':
                self.cams[cam_name] = RBRSCamera(device_id1=None, device_id2=f"/dev/{cam_name}_camera2")
            else:
                print(f"Camera {cam_name} does NOT exist !!")
                exit(1)

            try:
                self.cams[cam_name].start()
            except Exception as e:
                print(f"Error starting camera: {e}")
                exit(1)

        print("Camera recording started")

    def _check_proprio_reading(self):
        # Wait until all required observation keys are ready at least once
        test_dict = self.input_recorder.get_dict()
        obs_keys = [key for key in test_dict.keys() if not key.startswith("/action")]

        while True:
            d = self.input_recorder.get_dict()
            if all(d.get(k, None) is not None for k in obs_keys):
                break
            print("Waiting for proprio data to come in...")
            time.sleep(0.1)

        print("Proprio state recording started")

    
    def _spin_thread(self,):
        """Run rclpy executor in a background thread to service subscriptions."""
        try:
            self.executor.spin()
        except Exception as e:
            print(f"Error in spin thread: {e}")
    
    def _ensure_array_shape(self, arr, expected_shape):
        """
        Return arr as np.float32 with exact expected shape.
        If arr is None or has mismatched shape, return zeros.
        """
        if arr is None:
            return np.zeros(expected_shape, dtype=np.float32)
        arr = np.array(arr, dtype=np.float32)
        if arr.shape != expected_shape:
            return np.zeros(expected_shape, dtype=np.float32)
        return arr

    def _obs_dict_to_np_array(self, obs_dict: dict[str, np.ndarray]):
        """
        Convert observation dict to flat numpy array.

        For IGRIS_B, the state keys are predefined in IGRIS_B_STATE_KEYS.
        Each key maps to a 6D joint vector (position or current).
        Total: 8 keys * 6D = 48D state vector.
        """
        arrays = []
        for key in IGRIS_B_STATE_KEYS:
            if key in obs_dict and obs_dict[key] is not None:
                # Each observation is expected to be a 6D vector
                arrays.append(self._ensure_array_shape(obs_dict[key], (6,)))
            else:
                # Missing field → zero-fill with expected shape
                arrays.append(np.zeros((6,), dtype=np.float32))
        return np.concatenate(arrays, axis=-1)
    
    def shutdown(self):
        self.executor.shutdown()
        self.input_recorder.destroy_node()
        rclpy.shutdown()