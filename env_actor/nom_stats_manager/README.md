# env_actor/nom_stats_manager

Data normalization using dataset statistics. This module provides a **numpy-only** interface for normalizing robot observations and denormalizing policy actions. It has no torch dependency.

## How It Fits in the Pipeline

The normalization manager is instantiated by the inference loop and passed into the policy at every inference call. The policy calls `normalize_state()` internally before running the forward pass:

```
raw numpy obs → DataNormalizationInterface.normalize_state() → normalized numpy → policy forward pass
```

This ensures that normalization happens **inside** the policy boundary while keeping the normalizer itself free of torch.

## Interface / Bridge Pattern

[`data_normalization_interface.py`](data_normalization_interface.py) dispatches to robot-specific bridges:

```
DataNormalizationInterface(robot="igris_b", data_stats=stats)
    └── robots/igris_b/data_normalization_manager.py → DataNormalizationBridge
```

## Normalization Logic

The [`DataNormalizationBridge`](robots/igris_b/data_normalization_manager.py) applies:

**Proprioceptive state:**
```
normalized = (value - mean) / (std + eps)
```
where `eps = 1e-8` and `mean`/`std` are concatenated from `observation.state` and `observation.current` keys in the stats dict.

**Camera images:**
```
normalized = value / 255.0
```
Converts `uint8` images to `[0, 1]` float range.

**Action denormalization:**
```
denormalized = normalized_action * std + mean
```

## Stats Dictionary Structure

Loaded from a pickle file (path configured in `RuntimeParams.norm_stats_file_path`):

```python
{
    "observation.state": {"mean": np.ndarray, "std": np.ndarray},
    "observation.current": {"mean": np.ndarray, "std": np.ndarray},
    "action": {"mean": np.ndarray, "std": np.ndarray},
}
```

## Key Methods

| Method | Input | Output |
|--------|-------|--------|
| `normalize_state(state)` | dict with `"proprio"` + camera keys | Same dict, normalized in-place |
| `normalize_action(action)` | dict with `"action"` key | Same dict, normalized in-place |
| `denormalize_action(action)` | numpy array | Denormalized numpy array |

## Files

| File | Purpose |
|------|---------|
| [`data_normalization_interface.py`](data_normalization_interface.py) | Robot-agnostic interface; dispatches to bridge |
| [`robots/igris_b/data_normalization_manager.py`](robots/igris_b/data_normalization_manager.py) | igris_b normalization implementation |

## Adding a New Robot

1. Create `robots/<robot>/data_normalization_manager.py` with a `DataNormalizationBridge` class.
2. Implement `normalize_state()`, `normalize_action()`, and `denormalize_action()`.
3. Add the import branch in `data_normalization_interface.py`.

## Invariants

- **Numpy only.** This module must not import or use torch.
- **Stats come from the trainer.** The pickle file is produced during dataset creation and loaded via `RuntimeParams.read_stats_file()`.
