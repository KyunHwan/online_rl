"""
IGRIS_C robot initialization parameters and constants.

TODO: Update these values based on actual IGRIS_C hardware specifications.
"""

import numpy as np

# TODO: Define initial joint positions for IGRIS_C (in degrees, converted to radians)
# Example format for dual-arm robot (adjust based on actual DOF):
# INIT_JOINT_LIST = [+20.0, +30.0, 0.0, -120.0, 0.0, 0.0,  # Right arm (6 DOF)
#                    -20.0, -30.0, 0.0, +120.0, 0.0, 0.0]  # Left arm (6 DOF)
INIT_JOINT_LIST = []  # TODO: Specify initial joint configuration

# Convert degrees to radians
INIT_JOINT = (
    np.array(INIT_JOINT_LIST, dtype=np.float32) * np.pi / 180.0
    if INIT_JOINT_LIST
    else np.array([], dtype=np.float32)
)

# TODO: Define initial hand/gripper positions
# Example format:
# INIT_HAND_LIST = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5,  # Right hand (6 fingers)
#                   1.0, 1.0, 1.0, 1.0, 1.0, 0.5]  # Left hand (6 fingers)
INIT_HAND_LIST = []  # TODO: Specify initial hand/gripper configuration

# TODO: Define observation state keys for IGRIS_C
# These keys map to ROS2 topics and define the structure of the proprioceptive state
# Example format (adjust based on actual sensors):
# IGRIS_C_STATE_KEYS = [
#     "/observation/joint_pos/left",
#     "/observation/joint_pos/right",
#     "/observation/hand_joint_pos/left",
#     "/observation/hand_joint_pos/right",
#     "/observation/joint_cur/left",
#     "/observation/joint_cur/right",
#     "/observation/hand_joint_cur/left",
#     "/observation/hand_joint_cur/right",
# ]
IGRIS_C_STATE_KEYS = []  # TODO: Specify observation keys

# TODO: Verify that len(IGRIS_C_STATE_KEYS) * 6 = expected proprio_state_dim
# Or adjust based on actual state dimensionality

"""
HARDWARE SPECIFICATION CHECKLIST:

1. Joint Configuration:
   - [ ] Number of arm joints per side: ___
   - [ ] Number of hand/finger joints per side: ___
   - [ ] Total DOF: ___
   - [ ] Action dimensionality: ___

2. Sensors:
   - [ ] Joint position sensors
   - [ ] Joint current/torque sensors
   - [ ] Hand position sensors
   - [ ] Other sensors (IMU, F/T, etc.): ___

3. Initial Configuration:
   - [ ] Safe initial joint angles defined
   - [ ] Safe initial hand positions defined
   - [ ] Verified no self-collision in initial pose

4. State Space:
   - [ ] Total proprioceptive state dimension: ___
   - [ ] State keys documented
   - [ ] Each key maps to correct ROS2 topic
"""
