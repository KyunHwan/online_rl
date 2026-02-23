# env_actor/runtime_settings_configs

Robot-specific runtime parameters and configuration files. These define the physical and operational characteristics of each robot: sensor dimensions, control frequencies, action spaces, and communication topics.

## Structure

```
runtime_settings_configs/
├── robots/
│   ├── igris_b/
│   │   ├── inference_runtime_params.py      # RuntimeParams class
│   │   ├── inference_runtime_params.json    # Parameter values
│   │   ├── inference_runtime_topics.json    # ROS topic names
│   │   └── init_params.py                  # Initial robot position
│   └── igris_c/
│       ├── init_params.py
│       └── inference_runtime_params.py
└── __init__.py
```

## RuntimeParams

[`robots/igris_b/inference_runtime_params.py`](robots/igris_b/inference_runtime_params.py) defines the `RuntimeParams` class, which parses a JSON config dict and exposes typed properties:

| Property | Type | Description |
|----------|------|-------------|
| `HZ` | int | Control loop frequency (Hz) |
| `policy_update_period` | int | Steps between policy updates |
| `max_delta` | float | Maximum joint angle change per step (radians, from degrees) |
| `proprio_state_dim` | int | Dimensionality of proprioceptive state |
| `proprio_history_size` | int | Number of past observations to keep |
| `camera_names` | list | Names of camera streams |
| `num_img_obs` | int | Number of image observations to stack |
| `img_obs_every` | int | Subsampling rate for images |
| `mono_img_resize_width` | int | Image width after resize |
| `mono_img_resize_height` | int | Image height after resize |
| `action_dim` | int | Dimensionality of the action space |
| `action_chunk_size` | int | Number of timesteps per action chunk |

**`read_stats_file()`** loads normalization statistics from the pickle file at `norm_stats_file_path` (used by [`DataNormalizationInterface`](../nom_stats_manager/data_normalization_interface.py)).

## JSON Config Keys

The `inference_runtime_params.json` file contains:

```json
{
  "HZ": 50,
  "policy_update_period": 1,
  "max_delta_deg": 5.0,
  "proprio_state_dim": 32,
  "proprio_history_size": 1,
  "camera_names": ["head", "left", "right"],
  "num_img_obs": 1,
  "img_obs_every": 1,
  "mono_image_resize": {"width": 224, "height": 224},
  "action_dim": 32,
  "action_chunk_size": 40,
  "norm_stats_file_path": "/path/to/dataset_stats.pkl"
}
```

The `inference_runtime_topics.json` file maps logical names to ROS topic strings for robot communication.

## init_params.py

Contains the initial robot position (joint angles) used by `ControllerInterface.init_robot_position()` at the start of each episode.

## Usage

RuntimeParams is loaded in the inference and control loops:

```python
from env_actor.runtime_settings_configs.robots.igris_b.inference_runtime_params import RuntimeParams

runtime_params = RuntimeParams(json_config_dict)
print(runtime_params.action_chunk_size)  # 40
norm_stats = runtime_params.read_stats_file()  # dict for DataNormalizationInterface
```

## Adding a New Robot

1. Create `robots/<robot>/inference_runtime_params.py` with a `RuntimeParams` class.
2. Create the corresponding JSON config files.
3. Create `robots/<robot>/init_params.py` with the robot's home position.
