# Sequential Inference Refactoring - Summary

## Architecture Overview

The refactoring implements a clean separation of concerns:

```
┌─────────────────────────────────────────────────┐
│  CONTROLLER (I/O Only)                          │
│  - read_state() → raw observations              │
│  - publish_action(action) → robot               │
└─────────────────────────────────────────────────┘
                     ↓ raw obs
┌─────────────────────────────────────────────────┐
│  DATA MANAGER (ALL Processing)                  │
│  - Normalization stats & dimensions             │
│  - Observation history management               │
│  - Normalize observations                       │
│  - Denormalize actions                          │
│  - Action chunk buffering & selection           │
└─────────────────────────────────────────────────┘
                     ↓ normalized obs
┌─────────────────────────────────────────────────┐
│  POLICY (Neural Network Only)                   │
│  - Dictionary of nn.Module components           │
│  - forward(obs_dict) → normalized actions       │
│  - NO stats, NO dimensions, NO processing       │
└─────────────────────────────────────────────────┘
                     ↓ normalized actions
┌─────────────────────────────────────────────────┐
│  DATA MANAGER (Denormalization)                 │
│  - denormalize_action()                         │
│  - buffer_action_chunk()                        │
│  - get_current_action()                         │
└─────────────────────────────────────────────────┘
                     ↓ denormalized action
┌─────────────────────────────────────────────────┐
│  CONTROLLER (Publishing)                        │
│  - Slew-rate limiting                           │
│  - Publish to robot                             │
└─────────────────────────────────────────────────┘
```

## Key Changes

### 1. Policy (Simplified)
**Before**: Policy contained stats, dimensions, normalization logic
**After**: Policy is ONLY neural network forward pass

```python
# policy_manager.py
class PolicyManager:
    def __init__(self, policy_yaml_path, device):
        # Load policy using env_actor's build_policy
        self.policy = build_policy(policy_yaml_path, map_location=device)

    def predict(self, obs_dict):
        # Just forward pass - observation in, action out
        return self.policy.act(**obs_dict)  # or self.policy(**obs_dict)
```

### 2. Data Manager (Expanded)
**Before**: Partial implementation with broken normalization
**After**: Owns ALL data processing

```python
# data_manager_bridge.py (IGRIS_B)
class DataManagerBridge:
    def __init__(self, inference_runtime_config):
        # Load normalization stats from file
        self.norm_stats = self.runtime_params.read_stats_file()

        # Extract dimensions from stats/config
        self.state_dim = ...
        self.action_dim = ...
        self.num_queries = ...
        self.camera_names = ...

    # Processing methods
    def serve_normalized_obs_state(self, device) -> dict:
        """Normalize observations for policy."""

    def denormalize_action(self, action, device) -> np.ndarray:
        """Denormalize policy output."""

    def buffer_action_chunk(self, policy_output, current_step, device):
        """Buffer action chunk."""

    def get_current_action(self, current_step) -> np.ndarray:
        """Get action via simple indexing."""

    def generate_noise(self, device) -> torch.Tensor:
        """Generate noise for policy."""

    # Dimension properties
    @property
    def policy_state_dim(self) -> int:
        return self.state_dim
```

### 3. Sequential Engine (Simplified)
**Before**: Created observation/action processors, knew about stats
**After**: Simple orchestration only

```python
# sequential_engine.py
class SequentialInferenceEngine:
    def run(self):
        for t in range(self.max_timesteps):
            # 1. Read raw state from controller
            obs_data = self.controller_interface.read_state()

            # 2. Update history in data manager
            self.data_manager_interface.update_state_history(obs_data)

            # 3. Get normalized obs from data manager
            if (t % self.policy_update_period) == 0:
                normalized_obs = data_mgr.serve_normalized_obs_state(device)
                noise = data_mgr.generate_noise(device)
                normalized_obs['noise'] = noise

                # 4. Policy forward pass (just NN)
                policy_output = self.policy_manager.predict(normalized_obs)

                # 5. Denormalize and buffer in data manager
                data_mgr.buffer_action_chunk(policy_output, t, device)

            # 6. Get current action from data manager
            action = data_mgr.get_current_action(t)

            # 7. Publish to robot via controller
            smoothed = self.controller_interface.publish_action(action, prev_joint)
```

## Removed Files

- `observation_processor.py` - Logic moved to DataManager
- `action_processor.py` - Logic moved to DataManager
- `config_manager.py` - No longer needed

## Usage

```bash
python -m env_actor.auto.inference_algorithms.sequential.sequential_runner \
  --runtime_config path/to/runtime_config.json \
  --policy_yaml path/to/policy.yaml \
  --robot igris_b \
  --device cuda
```

## Configuration Format

The `runtime_config.json` should include:

```json
{
  "robot_id": "packy",
  "HZ": 20,
  "policy_update_period": 40,
  "sequential": {
    "max_timesteps": 1000,
    "max_delta": 0.01
  },
  "camera_names": ["head", "left", "right"],
  "proprio_state_dim": 48,
  "action_dim": 24,
  "action_chunk_size": 40,
  "proprio_history_size": 40,
  "num_img_obs": 1,
  "img_obs_every": 1,
  "mono_image_resize": {
    "width": 320,
    "height": 240
  },
  "norm_stats_file_path": "/path/to/stats.pkl",
  "topics": { ... }
}
```

The `stats.pkl` file should contain:

```python
{
    'state_mean': np.ndarray,      # (state_dim,)
    'state_std': np.ndarray,       # (state_dim,)
    'action_mean': np.ndarray,     # (action_dim,)
    'action_std': np.ndarray,      # (action_dim,)
    'dimensions': {                # Optional - falls back to config
        'state_dim': int,
        'action_dim': int,
        'num_queries': int,
        'num_robot_obs': int,
        'num_image_obs': int,
    }
}
```

## Benefits

1. **Clear Separation**: Policy knows nothing about data processing
2. **Modularity**: DataManager is reusable for RTC
3. **No Dependency on inference_engine**: Uses only env_actor components
4. **Testability**: Each component has clear, focused responsibilities
5. **Maintainability**: Changes to processing don't affect policy

## Migration Guide

If you have existing code using the old architecture:

**Before:**
```python
from env_actor.auto.inference_algorithms.shared import (
    ObservationProcessor, ActionProcessor, ConfigManager
)

policy_manager = PolicyManager(model_config, policy_config, checkpoint, device)
obs_processor = ObservationProcessor(policy_manager.state_mean, ...)
action_processor = ActionProcessor(policy_manager.action_mean, ...)
```

**After:**
```python
# Policy is simple
policy_manager = PolicyManager(policy_yaml_path, device)

# DataManager owns all processing
data_manager = DataManagerInterface(runtime_config, robot)

# All dimensions come from data manager
print(data_manager.data_manager_bridge.policy_state_dim)
```

## Testing

The refactored code maintains the same sequential inference logic but with cleaner separation:

1. **Policy**: Test neural network forward pass independently
2. **DataManager**: Test normalization math, history management
3. **Controller**: Test I/O with mock ROS2
4. **SequentialEngine**: Test orchestration with mocks

## Next Steps

1. Test with actual robot hardware
2. Implement RTC using the same DataManager
3. Add IGRIS_C support by implementing bridges
4. Remove any remaining dependencies on `inference_engine` directory
