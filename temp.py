#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState

import numpy as np
import argparse

import time
import threading
import signal
import sys
from contextlib import nullcontext

from engine.algorithms.utils.data_dict import GenericRecorder
from engine.algorithms.utils.camera_utils import RBRSCamera
from engine.policies.loader import build_policy
from engine.policies.interfaces import InferencePolicy

import torch
import cv2

from engine.algorithms.config.config_loader import ConfigLoader
from engine.config.inference_schemas import ModelConfig, PolicyConfig
from engine.registry.plugins import load_plugins

# Initial positions (deg → rad), initial finger targets, and initial wait time
INIT_JOINT = np.array(
    [+20.0,+30.0,0.0,-120.0,0.0,0.0,-20.0,-30.0,0.0,+120.0,0.0,0.0],
    dtype=np.float32
) * np.pi / 180.0
INIT_TIME  = 0.0

IGRIS_STATE_KEYS = [
    "/observation/joint_pos/left",
    "/observation/joint_pos/right",
    "/observation/hand_joint_pos/left",
    "/observation/hand_joint_pos/right",
    "/observation/joint_cur/left",
    "/observation/joint_cur/right",
    "/observation/hand_joint_cur/left",
    "/observation/hand_joint_cur/right",
]

def signal_handler(signum, frame):
    """Trap SIGINT/SIGTERM for graceful shutdown."""
    print(f"\nReceived signal {signum}. Saving data and shutting down gracefully...")
    sys.exit(0)

def ensure_array_shape(arr, expected_shape):
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

def obs_dict_to_np_array(obs_dict: dict[str, np.ndarray], config: ConfigLoader) -> np.ndarray:
    """
    Flatten a structured observation dict into a single 1D state vector.
    Field layout & slicing is driven by the runtime config.
    Missing values are zero-filled to preserve shape and ordering.
    """
    expected_keys = config.get_observation_keys()
    arrays = []
    for key in IGRIS_STATE_KEYS:
        field_config = config.get_observation_field_config(key)
        if key in obs_dict and obs_dict[key] is not None:

            if key == "/observation/barcode":
                arrays.append(np.array([1.0 if obs_dict[key] else 0.0], dtype=np.float32))
            elif field_config == "pose.position":
                arrays.append(ensure_array_shape(obs_dict[key], (3,)))
            elif field_config == "pose.orientation":
                arrays.append(ensure_array_shape(obs_dict[key], (4,)))
            elif isinstance(field_config, dict) and "slice" in field_config:
                s = field_config["slice"]
                arrays.append(ensure_array_shape(obs_dict[key], (s[1] - s[0],)))
            else:
                # Default structural field size: 6 (e.g., twist or pose6D)
                arrays.append(ensure_array_shape(obs_dict[key], (6,)))
        else:
            # Missing field → zero-fill with the expected shape
            if key == "/observation/barcode":
                arrays.append(np.array([0.0], dtype=np.float32))
            elif field_config == "pose.position":
                arrays.append(np.zeros((3,), dtype=np.float32))
            elif field_config == "pose.orientation":
                arrays.append(np.zeros((4,), dtype=np.float32))
            elif isinstance(field_config, dict) and "slice" in field_config:
                s = field_config["slice"]
                arrays.append(np.zeros((s[1] - s[0],), dtype=np.float32))
            else:
                arrays.append(np.zeros((6,), dtype=np.float32))
    return np.concatenate(arrays, axis=-1)

def spin_thread(executer: SingleThreadedExecutor):
    """Run rclpy executor in a background thread to service subscriptions."""
    try:
        executer.spin()
    except Exception as e:
        print(f"Error in spin thread: {e}")


def main(args):
    # Install signal handlers early
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Let cuDNN search fast kernels dynamically
    torch.backends.cudnn.benchmark = True
    #load_plugins(args.get("plugins", [])) # already being done in inference.py

    # --- Load runtime config & resolve runtime parameters ---
    # config       = ConfigLoader(args["runtime_config_path"])
    # task_config  = config.get_config()
    # inference_settings = config.get_inference_settings("sequential")

    # max_timesteps = int(inference_settings["max_timesteps"])
    # robot_id     = task_config["robot_id"]
    # camera_names = config.get_camera_names()
    #camera_names = ["left", "right", "head"]  
    # print(camera_names)
    
    esb_k               = float(inference_settings.get("esb_k", 0.01))                  # temporal ensemble coefficient
    policy_update_period= max(1, int(inference_settings.get("policy_update_period", 1)))
    temporal_ensemble   = bool(inference_settings.get("temporal_ensemble", False))
    HZ_override = args.get("hz")
    HZ = task_config["HZ"] if HZ_override is None else HZ_override
    max_delta_deg       = float(inference_settings.get("max_delta", 10.0))              # input is degrees
    image_width, image_height = config.get_image_resize()
    # print(f"policy_update_period: {policy_update_period}")
    # print(f"temporal_ensemble: {temporal_ensemble}")
    # print(f"HZ: {HZ}")
    # print()
    
    # # --- ROS2 publishers & executor ---
    # input_recorder = GenericRecorder(task_config)  # node providing observations/images

    # qos = QoSProfile(depth=10)
    # qos.reliability = QoSReliabilityPolicy.RELIABLE

    # joint_pub = input_recorder.create_publisher(JointState,
    #                 f"/igris_b/{robot_id}/target_joints", qos_profile=qos)
    # finger_pub = input_recorder.create_publisher(Float32MultiArray,
    #                 f"/igris_b/{robot_id}/finger_target", qos_profile=qos)
    # # stop_publisher = input_recorder.create_publisher(Bool,
    # #                 f"/igris_b/{robot_id}/stop", qos_profile=qos)

    # executor = SingleThreadedExecutor()
    # executor.add_node(input_recorder)
    # thread = threading.Thread(target=spin_thread, args=(executor,), daemon=True)
    # thread.start()

    # # --- Camera setup --- #
    # head_cam = RBRSCamera(device_id1="/dev/head_camera1", device_id2=None)
    # left_cam = RBRSCamera(device_id1=None, device_id2="/dev/left_camera2")
    # right_cam = RBRSCamera(device_id1="/dev/right_camera1", device_id2=None)

    # cams = {
    #     'head': head_cam, 
    #     'left': left_cam, 
    #     'right': right_cam
    # }
    
    # for camera_name in camera_names:
    #     try:
    #         cams[camera_name].start()
    #     except Exception as e:
    #         print(f"Error starting camera: {e}")
    #         exit(1)

    try:
        # --- Load policy checkpoint & dataset stats (CUDA only) ---
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("high")

        model_cfg = ModelConfig.model_validate(args["model"])
        policy_cfg = PolicyConfig.model_validate(args["policy"])
        policy = build_policy(
            model_cfg,
            policy_cfg,
            args["checkpoint_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.eval()
        print("Loaded policy via policy builder")

        if set(camera_names) != set(policy.camera_names):
            raise ValueError("Runtime camera_names does not match policy camera_names")

        sm, ss, am, asd, eps = policy.normalization_tensors
        device = sm.device
        

        # --- Unpack model I/O dimensions and image cadence ---
        state_dim      = policy.state_dim
        action_dim     = policy.action_dim
        num_queries    = policy.num_queries
        num_robot_obs  = policy.num_robot_observations
        num_image_obs  = policy.num_image_observations
        image_obs_every_setting = inference_settings.get("image_obs_every", 1)
        image_obs_every= image_obs_every_setting if image_obs_every_setting != -1 else policy.image_observation_skip

        # Per-camera image history buffer: N x C x H x W (uint8)
        # image_obs_history = {
        #     cam: np.zeros((num_image_obs, 3, image_height, image_width // 2), dtype=np.uint8)
        #     for cam in camera_names
        # }
        # image_frame_counter = 0

        # # Wait until all required observation keys are ready at least once
        # test_dict = input_recorder.get_dict()
        # obs_keys = [key for key in test_dict.keys() if not key.startswith("/action")]

        # while True:
        #     d = input_recorder.get_dict()
        #     if all(d.get(k, None) is not None for k in obs_keys):
        #         break
        #     print("Proprio data not coming in...")
        #     time.sleep(0.1)

        # # Deassert stop and optionally publish initial posture (kept commented by design)
        # # stop_publisher.publish(Bool(data=False))
        # print("Start published")

        

        # Build initial robot state vector & bootstrap history with it
        # obs_dict = input_recorder.get_observation_dict()
        # robot_state = obs_dict_to_np_array(obs_dict, config)
        # if robot_state.shape[0] != state_dim:
        #     raise ValueError(f'not match robot state shape({robot_state.shape}) and state_dim({state_dim})')

        # robot_obs_history = np.zeros((num_robot_obs, state_dim), dtype=np.float32)
        # robot_obs_history[:] = np.repeat(robot_state.reshape(1, -1), num_robot_obs, axis=0)

        # Optional initial wait to align timing with external systems
        init_duration = 0.0  # small book-keeping if needed later
        wait_duration = INIT_TIME - init_duration
        if wait_duration > 0.0:
            time.sleep(wait_duration)

        # --- Control loop timing --- #
        rate = input_recorder.create_rate(HZ)
        DT   = 1.0 / HZ
        start_time = time.time()
        next_t = time.perf_counter()

        # --- Motion smoothing setup ---
        # prev_joint = INIT_JOINT.copy()
        # joint_msg = JointState()
        # joint_msg.position = prev_joint
        # joint_pub.publish(joint_msg)
        
        #max_delta = np.deg2rad(max_delta_deg)  # convert degrees → radians once
        #print(f"max_delta: {max_delta}")
        # Buffer for temporal fusion of overlapping action sequences
        all_time_actions = np.zeros((max_timesteps, max_timesteps + num_queries, action_dim), dtype=np.float32)
        last_all_actions = None      # last policy output (torch tensor)
        last_policy_update_step = -1
        # Operator gate to start (avoid accidental motion)
        
        input("Press any key to start inference...")
        #temporal_ensemble=False
        # =========================
        #   MAIN CONTROL LOOP
        # =========================
        for t in range(max_timesteps):
            # rate.sleep()  # block to maintain target HZ

            # # --- Read latest inputs ---
            # obs_dict   = input_recorder.get_observation_dict()

            # # Pack/validate robot state
            # try:
            #     robot_state = obs_dict_to_np_array(obs_dict, config)
            # except Exception as e:
            #     print(f"Error getting robot state at timestep {t}: {str(e)}")
            #     continue

            # Update robot state history (most recent at index 0)
            # if num_robot_obs > 1:
            #     robot_obs_history[1:] = robot_obs_history[:-1]
            # # print(f"robot_state shape: {robot_state.shape}")
            # # print(f"robot_state dtype: {robot_state.dtype}")
            # robot_obs_history[0] = robot_state

            # Normalize state history for model input
            # rh = torch.from_numpy(robot_obs_history).to(device=device, dtype=torch.float32, non_blocking=True)
            # if sm.numel() == state_dim and ss.numel() == state_dim:
            #     rh = (rh - sm.view(1, -1)) / (ss.view(1, -1) + eps)
            # else:
            #     # In case stats are flattened differently, repeat to match shape
            #     sm_rep = sm if sm.numel() == num_robot_obs * state_dim else sm.repeat((num_robot_obs * state_dim + sm.numel() - 1) // sm.numel())[:num_robot_obs * state_dim]
            #     ss_rep = ss if ss.numel() == num_robot_obs * state_dim else ss.repeat((num_robot_obs * state_dim + ss.numel() - 1) // ss.numel())[:num_robot_obs * state_dim]
            #     rh = (rh - sm_rep.view(num_robot_obs, state_dim)) / (ss_rep.view(num_robot_obs, state_dim) + eps)
            # rh = rh.reshape(1, -1).view(1, policy.num_robot_observations, policy.state_dim)

            # Maintain per-camera image history with downscaled frames
            # cam_list = []
            # # for cam_name in camera_names:
            # #     #image_key = f"/observation/images/{cam_name}"
            # #     cur = cams[cam_name].get_image()
            # #     if cur is not None:
            # #         cur = cv2.resize(
            # #             cur,
            # #             dsize=(image_width // 2, image_height),
            # #             interpolation=cv2.INTER_AREA,
            # #         )
            # #         #if cam_name == 'left' or cam_name == 'right': cv2.imwrite("image", cur)
            # #         cur = np.transpose(cur, (2, 0, 1))  # HWC → CHW
            #         if image_obs_every <= 1 or (image_frame_counter % image_obs_every == 0):
            #             if num_image_obs > 1:
            #                 image_obs_history[cam_name][1:] = image_obs_history[cam_name][:-1]
            #             image_obs_history[cam_name][0] = cur
            #     # else: 
            #     #     print(f"{cam_name}: image is None!!!!")
                        
            #     cam_list.append(image_obs_history[cam_name])
            # image_frame_counter += 1

            if len(cam_list) == 0:
                continue

            # Aggregate all cameras: [num_cams, Timg, C, H, W] → batched float16
            all_cam_images = np.stack(cam_list, axis=0)
            
            # print(f"all_cam_images shape: {all_cam_images.shape}")
            # print(f"all_cam_images dtype: {all_cam_images.dtype}")
            
            img_dtype = torch.float16 if device.type == "cuda" else torch.float32
            cam_images = torch.from_numpy(all_cam_images / 255.0).to(
                device=device, dtype=img_dtype, non_blocking=True
            ).unsqueeze(0)
            
            
            # Policy update: either per-step or re-use last prediction over horizon
            if (t % policy_update_period) == 0 or last_all_actions is None:
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_ctx:
                    new_actions = policy(rh, cam_images, noise=torch.randn(1, num_queries, action_dim).to(device))  # shape: (1, num_queries, action_dim)
                    # De-normalize actions
                   
                    am_rep  = am  if am.numel()  == action_dim else am.repeat((action_dim + am.numel() - 1) // am.numel())[:action_dim]
                    asd_rep = asd if asd.numel() == action_dim else asd.repeat((action_dim + asd.numel() - 1) // asd.numel())[:action_dim]
                    new_actions = (new_actions * asd_rep.view(1, 1, -1) + am_rep.view(1, 1, -1))
                last_all_actions = new_actions.detach()
                last_policy_update_step = t
            
            # Copy into temporal fusion buffer
            aa = last_all_actions.float().cpu().numpy().squeeze(0)  # (num_queries, action_dim)
            all_time_actions[t, t:t + num_queries, :] = aa[:]

            # Collect all predictions that target current time index t
            # action_bottom_idx = max(t - num_queries + 1, 0)
            # actions_for_curr_step = all_time_actions[action_bottom_idx:(t + 1), t, :]

            # Choose current action: ensemble or index into rolling horizon
            # if temporal_ensemble:
            #     # Exponential weights favor most recent prediction
            #     exp_weights = np.exp(esb_k * np.arange(len(actions_for_curr_step), dtype=np.float32))
            #     exp_weights = exp_weights / exp_weights.sum()
            #     exp_weights = exp_weights.reshape(-1, 1)
            #     action = (actions_for_curr_step * exp_weights).sum(axis=0)
            else:
                offset = t - last_policy_update_step
                idx = int(np.clip(offset, 0, aa.shape[0] - 1))
                action = aa[idx]
            
            # Split action vector into joint/finger segments
            # left_joint_pos   = action[:6]
            # right_joint_pos  = action[6:12]
            # left_finger_pos  = action[12:18]
            # right_finger_pos = action[18:24]

            # # Merge to robot joint order (environment-specific; verify mapping)
            # raw_joint = np.concatenate([right_joint_pos, left_joint_pos])

            # # Slew-rate limiting per joint (rad/step)
            # delta = np.clip(raw_joint - prev_joint, -max_delta, max_delta)
            # #print(f"raw_joint: {raw_joint}\n")
            # #print(f"delta: {delta}\n")
            # smoothed = prev_joint + delta
            # prev_joint = smoothed

            # # Publish joints
            # joint_msg = JointState()
            # joint_msg.position = smoothed.tolist()
            # joint_pub.publish(joint_msg)

            # # Publish fingers (right followed by left to match action layout)
            # finger_msg = Float32MultiArray()
            # finger_msg.data = list(right_finger_pos) + list(left_finger_pos)
            # finger_pub.publish(finger_msg)
            #print(f"fingers: {list(right_finger_pos) + list(left_finger_pos)}")
            # Maintain precise loop timing against drift
            next_t += DT
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)

            # Periodic health print
            if (t % max(1, HZ * 2) == 0) and (t > 0):
                print(f"Avg Control Frequency [Hz]: {t / (time.time() - start_time)} steps: {t}/{max_timesteps}")

    except Exception as e:
        # Log and proceed to finally for cleanup
        print(e)
    finally:
        # Best-effort stop and teardown
        try:
            # stop_publisher.publish(Bool(data=True))
            pass
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            input_recorder.destroy_node()
        except Exception:
            pass
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass

if __name__ == "__main__":
    # Parse args early and start ROS
    parser = argparse.ArgumentParser()
    rclpy.init()

    parser.add_argument("--inference_config", "-I", type=str, required=True,
                        help="Path to inference YAML (validated schema)")
    from engine.config.loader import load_config
    from engine.config.inference_schemas import validate_inference_config
    args = parser.parse_args()
    raw = load_config(args.inference_config)
    cfg = validate_inference_config(raw)
    main(
        {
            "runtime_config_path": cfg.runtime_config_path,
            "checkpoint_path": cfg.checkpoint_path,
            "model": cfg.model.model_dump(),
            "policy": cfg.policy.model_dump(),
            "plugins": list(cfg.plugins),
            "hz": cfg.hz if cfg.hz is not None else None,
        }
    )
