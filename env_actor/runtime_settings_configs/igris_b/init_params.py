import numpy as np

# Initial positions (deg → rad) and initial finger targets
INIT_JOINT_LIST = [+20.0,+30.0,0.0,-120.0,0.0,0.0, -20.0,-30.0,0.0,+120.0,0.0,0.0] # in right and left order
INIT_HAND_LIST = [1.0,1.0,1.0,1.0,1.0,0.5, 1.0,1.0,1.0,1.0,1.0,0.5]
INIT_JOINT = np.array(
    INIT_JOINT_LIST,
    dtype=np.float32
) * np.pi / 180.0

IGRIS_B_STATE_KEYS = [
    "/observation/joint_pos/left",
    "/observation/joint_pos/right",
    "/observation/hand_joint_pos/left",
    "/observation/hand_joint_pos/right",
    # "/observation/joint_cur/left",
    # "/observation/joint_cur/right",
    # "/observation/hand_joint_cur/left",
    # "/observation/hand_joint_cur/right",
]

