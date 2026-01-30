# Registry

Registries map string keys to component classes and are the backbone of configuration-driven instantiation.

Modules:
- `core.py`: simple registry with optional base-class enforcement.
- `builtins.py`: registers built-in trainers, data, and components.
- `plugins.py`: loads user modules listed in `plugins`.

Common registries (see `trainer/registry/__init__.py`):
- `TRAINER_REGISTRY`
- `DATAMODULE_REGISTRY`
- `OPTIMIZER_REGISTRY`
- `SCHEDULER_REGISTRY`
- `LOSS_REGISTRY`
- `METRICS_REGISTRY`
- `CALLBACK_REGISTRY`
- `LOGGER_REGISTRY`

Register a custom component with a decorator:
```
@DATAMODULE_REGISTRY.register("my_data")
class MyDataModule:
    ...
```
