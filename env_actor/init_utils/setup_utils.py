import numpy as np
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

# Initial positions (deg → rad) and initial finger targets
INIT_JOINT_LIST = [+20.0,+30.0,0.0,-120.0,0.0,0.0, -20.0,-30.0,0.0,+120.0,0.0,0.0] # in right and left order
INIT_HAND_LIST = [1.0,1.0,1.0,1.0,1.0,0.5, 1.0,1.0,1.0,1.0,1.0,0.5]
INIT_JOINT = np.array(
    INIT_JOINT_LIST,
    dtype=np.float32
) * np.pi / 180.0

@dataclass(frozen=True)
class ShmArraySpec:
    name: str
    shape: tuple
    dtype_str: str  # e.g., np.float32().dtype.str

def init_shared_action_chunk(num_rows: int = 40, dtype=np.float32):
    # 24-D vector: [L arm 6] + [L hand 6] + [R arm 6] + [R hand 6]
    init_vec = np.asarray(
        INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6] + INIT_HAND_LIST[:6] + INIT_HAND_LIST[6:],
        dtype=dtype,
    )  # shape (24,)
    init_vec[:12] *= (np.pi / 180.0)
    init_vec[12:] *= 0.03

    # Repeat across rows -> (40, 24)
    action_chunk = np.tile(init_vec, (num_rows, 1))  # writable, contiguous

    return action_chunk

# used in child process
def attach_shared_ndarray(spec: ShmArraySpec, unregister: bool = True):
    shm = SharedMemory(name=spec.name)
    if unregister:
        # Let the parent manage unlinking; avoid auto-unlink from the child's resource tracker.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype_str), buffer=shm.buf)
    return shm, arr