from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker
import numpy as np

@dataclass(frozen=True)
class ShmArraySpec:
    """Specification for attaching to a SharedMemory array by name.

    Attributes:
        name: The OS-level shared memory name
        shape: Shape of the numpy array
        dtype_str: Numpy dtype string (e.g., np.float32().dtype.str)
    """

    name: str
    shape: tuple
    dtype_str: str

def create_shared_ndarray(
    shape: tuple, dtype: np.dtype, zero: bool = True
) -> tuple[SharedMemory, np.ndarray, ShmArraySpec]:
    """Create a SharedMemory block and wrap it as a numpy array.

    Args:
        shape: Shape of the array
        dtype: Numpy dtype for the array
        zero: Whether to zero-initialize the array

    Returns:
        Tuple of (SharedMemory handle, numpy array view, ShmArraySpec for sharing)
    """
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if zero:
        arr[...] = 0
    spec = ShmArraySpec(name=shm.name, shape=shape, dtype_str=dtype.str)
    return shm, arr, spec


def attach_shared_ndarray(
    shared_spec_dict: dict[str, ShmArraySpec], unregister: bool = True
) -> tuple[SharedMemory, np.ndarray]:
    """Attach to an existing SharedMemory block by name.

    Args:
        spec: ShmArraySpec with name, shape, and dtype info
        unregister: If True, unregister from resource tracker to prevent
                    auto-unlink by child processes (parent handles cleanup)

    Returns:
        Tuple of (SharedMemory handle, numpy array view)
    """
    shm_handles = {}
    shm_arr = {}
    for key in shared_spec_dict.keys():
        spec = shared_spec_dict[key]
        shm_handles[key] = SharedMemory(name=spec.name)
        if unregister:
            # Let the parent manage unlinking; avoid auto-unlink from the child's resource tracker.
            try:
                resource_tracker.unregister(shm_handles[key]._name, "shared_memory")
            except Exception:
                pass
        shm_arr[key] = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype_str), buffer=shm_handles[key].buf)
    return shm_handles, shm_arr