# IGRIS_C Robot Interface Documentation

This directory contains interface stubs and documentation for integrating the IGRIS_C robot platform.

## Status

**Interface Design Only** - Full implementation deferred until hardware specifications are available.

## Overview

The IGRIS_C integration follows the same modular pattern as IGRIS_B:
- `controller_bridge.py`: Robot-specific I/O (ROS2, cameras, action publishing)
- `data_manager_bridge.py`: State history and normalization management
- Runtime configuration in `runtime_settings_configs/igris_c/`

## Required Hardware Specifications

Before implementing IGRIS_C, the following hardware specifications must be determined:

### 1. Robot Kinematic Structure
- **Number of DOF**: How many joints? (IGRIS_B has 12 arm joints)
- **Joint Layout**: Dual-arm? Single-arm? Mobile base?
- **Hand/Gripper**: Number of finger joints? (IGRIS_B has 6 per hand)
- **Action Dimension**: Total action space dimensionality

### 2. Proprioceptive State Structure
- **State Keys**: What observations are available?
  - Joint positions (left/right)
  - Joint currents/torques
  - Hand/finger positions
  - End-effector poses
  - Other sensor readings (IMU, force/torque, etc.)
- **State Dimension**: Total proprioceptive state dimensionality

### 3. ROS2 Topic Configuration
- **Topic Naming Convention**: 
  - Does IGRIS_C use `/igris_c/{robot_id}/...` pattern?
  - Or a different naming scheme?
- **Required Topics**:
  - Joint state publishers/subscribers
  - Finger/gripper control topics
  - Image topics (if camera mounted on robot)
  - Command topics for action execution

### 4. Camera System
- **Camera Names**: What cameras are available? (e.g., head, left, right, wrist)
- **Device Mapping**: 
  - V4L2 device paths (e.g., `/dev/igris_c_head_camera`)
  - Camera API (V4L2, RealSense, custom?)
- **Resolution and Frame Rate**: Native and target resolutions
- **Image Processing**: Any robot-specific preprocessing required?

### 5. Control Interface
- **Control Frequency**: Target Hz (IGRIS_B uses 20Hz)
- **Action Format**: How are actions structured?
  - Joint space vs. task space
  - Absolute positions vs. deltas
  - Quaternions vs. Euler angles (if applicable)
- **Safety Limits**:
  - Maximum joint velocities
  - Joint position limits
  - Slew-rate limiting parameters

## Implementation Checklist

When hardware specs become available, implement in this order:

### Phase 1: Configuration
- [ ] Define `IGRIS_C_STATE_KEYS` in `init_params.py`
- [ ] Set `INIT_JOINT` and `INIT_HAND_LIST` initial positions
- [ ] Create `inference_runtime_settings.json` with topics and field mappings
- [ ] Implement `RuntimeParams` class with IGRIS_C-specific properties

### Phase 2: Controller Bridge
- [ ] Implement ROS2 node initialization
- [ ] Implement `read_state()` for proprioception reading
- [ ] Implement `publish_action()` for action execution
- [ ] Implement camera initialization and capture
- [ ] Add slew-rate limiting with IGRIS_C-specific limits
- [ ] Implement `_obs_dict_to_np_array()` for state packing

### Phase 3: Data Manager Bridge
- [ ] Implement observation history buffers
- [ ] Implement normalization with IGRIS_C stats
- [ ] Implement `serve_normalized_obs_state()`
- [ ] Implement `denormalize_action()`

### Phase 4: Testing
- [ ] Test ROS2 communication with mock topics
- [ ] Test camera capture and image processing
- [ ] Validate state dimensionality matches policy expectations
- [ ] End-to-end integration test with sequential inference engine

## Reference Implementation: IGRIS_B

Use IGRIS_B as a reference template:
- Controller Bridge: `/env_actor/auto/io_interface/robots/igris_b/controller_bridge.py`
- Data Manager Bridge: `/env_actor/auto/data_manager/robots/igris_b/data_manager_bridge.py`
- Runtime Params: `/env_actor/runtime_settings_configs/robots/igris_b/`

### Key Differences to Consider

IGRIS_C may differ from IGRIS_B in:
1. **State Dimensionality**: Different number of joints or sensors
2. **Camera Configuration**: Different camera setup or placement
3. **ROS2 Topics**: Different topic structure or message types
4. **Action Space**: Different control interface or action format
5. **Safety Constraints**: Different joint limits or velocity constraints

## Example: Adapting from IGRIS_B

If IGRIS_C is similar to IGRIS_B but with different joint counts:

```python
# init_params.py
import numpy as np

# TODO: Update based on IGRIS_C hardware
INIT_JOINT_LIST = [...]  # Initial joint positions in degrees
INIT_JOINT = np.array(INIT_JOINT_LIST, dtype=np.float32) * np.pi / 180.0

# TODO: Define IGRIS_C state observation keys
IGRIS_C_STATE_KEYS = [
    "/observation/joint_pos/...",
    # Add all observation keys
]
```

## Contact

For questions about IGRIS_C integration or to provide hardware specifications:
- Document specifications in this directory
- Update the implementation checklist as components are completed
- Reference the IGRIS_B implementation for guidance
