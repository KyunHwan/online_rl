import numpy as np
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from env_actor.runtime_settings_configs.igris_b.init_params import (
    INIT_JOINT_LIST,
    INIT_HAND_LIST,
    INIT_JOINT,
    IGRIS_B_STATE_KEYS
)

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