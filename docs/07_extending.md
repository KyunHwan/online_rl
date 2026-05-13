# 07 — Extending

Four recipes for plugging new components into the pipeline. Each recipe says exactly which files to add, which interfaces to satisfy, and where the dispatch happens.

If you have not yet read [05_policy_protocol.md](05_policy_protocol.md) and [08_invariants.md](08_invariants.md), do that first. Most extension mistakes are invariant violations.

## Table of contents

- [Recipe 1: Add a new policy](#recipe-1-add-a-new-policy)
- [Recipe 2: Add a new robot](#recipe-2-add-a-new-robot)
- [Recipe 3: Add a new inference algorithm](#recipe-3-add-a-new-inference-algorithm)
- [Recipe 4: Add a new reward labeler](#recipe-4-add-a-new-reward-labeler)

---

## Recipe 1: Add a new policy

You want to plug a new model architecture into the inference loop.

### Files to add

```text
env_actor/policy/policies/<your_policy>/
├── __init__.py
├── <your_policy>.py
├── <your_policy>.yaml
└── components/
    ├── __init__.py
    └── <component>.yaml      # one per nn.Module the factory should build
```

### Step 1 — write the policy class

```python
# env_actor/policy/policies/my_policy/my_policy.py
from typing import Any
import numpy as np
import torch
from torch import nn

from env_actor.policy.registry import POLICY_REGISTRY
from env_actor.inference_engine_utils.action_inpainting import compute_guided_prefix_weights


@POLICY_REGISTRY.register("my_policy")
class MyPolicy:
    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None:
        # MUST store as self.components — the trainer broadcasts weights using this attribute name.
        self.components = components
        # ... read any kwargs from the YAML's policy.params

    def predict(self, obs: dict, data_normalization_interface) -> np.ndarray:
        # numpy in, numpy out. (action_horizon, action_dim) float32.
        ...

    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed: int, action_chunk_size: int) -> np.ndarray:
        pred_actions = self._forward(input_data, data_normalization_interface)
        weights = compute_guided_prefix_weights(
            input_data["est_delay"], min_num_actions_executed, action_chunk_size, schedule="exp"
        ).reshape(-1, 1)
        return input_data["prev_action"] * weights + pred_actions * (1.0 - weights)

    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
    def eval(self): ...
    def to(self, device): ...
```

Don't forget the surrounding `.eval()` / `.to(device)` methods — [build_policy()](../env_actor/policy/utils/loader.py) calls both before handing the policy to the inference loop.

Full reference: [05_policy_protocol.md](05_policy_protocol.md).

### Step 2 — write the policy YAML

```yaml
# env_actor/policy/policies/my_policy/my_policy.yaml
model:
  component_config_paths:
    encoder: components/encoder.yaml
    decoder: components/decoder.yaml

policy:
  type: my_policy
  params:
    some_kwarg: 42
```

Component paths are resolved relative to this YAML's directory. The key names (`encoder`, `decoder`) become the keys of `self.components` and must match the trainer-side `model.component_build_args` keys for weight broadcasting to work.

### Step 3 — write the component YAMLs

Each component YAML is parsed by `PolicyConstructorModelFactory` from the [trainer submodule](../trainer/). Schema and examples are in [trainer/docs/05_configuration.md](../trainer/docs/05_configuration.md) and the existing components under [env_actor/policy/policies/dsrl_openpi_policy/components/](../env_actor/policy/policies/dsrl_openpi_policy/components/).

### Step 4 — run it

```bash
python run_online_rl.py \
  --policy_yaml ./env_actor/policy/policies/my_policy/my_policy.yaml \
  --train_config ./path/to/your/train_config.yaml
```

`build_policy()` will:

1. Read the YAML.
2. See `policy.type: my_policy` is not yet in `POLICY_REGISTRY`.
3. Run `importlib.import_module("env_actor.policy.policies.my_policy.my_policy")` — that triggers the `@POLICY_REGISTRY.register` decorator at import time.
4. Look up the class and instantiate it with the built components and any `policy.params`.

No other file in the pipeline needs to change.

---

## Recipe 2: Add a new robot

You want to support a new physical robot, say `igris_d`.

### Why this is a lot of files

The repo follows an **interface + per-robot bridge** pattern. Every robot-specific behavior lives behind an interface that dispatches on the `robot` string. So a new robot means writing a bridge in each interface's directory.

### Files to add

```text
env_actor/
├── robot_io_interface/robots/igris_d/
│   ├── __init__.py
│   ├── controller_bridge.py            # ROS2 publishers, camera readers
│   └── utils/                           # camera_utils, data_dict — optional
├── nom_stats_manager/robots/igris_d/
│   ├── __init__.py
│   └── data_normalization_manager.py   # numpy-only, per-robot norm logic
├── episode_recorder/robots/igris_d/
│   ├── __init__.py
│   └── episode_recorder_bridge.py      # builds TensorDicts from obs/action lists
├── auto/inference_algorithms/rtc/data_manager/robots/igris_d/
│   ├── __init__.py
│   └── shm_manager_bridge.py           # RTC shared-memory bridge
├── auto/inference_algorithms/sequential/data_manager/robots/igris_d/
│   ├── __init__.py
│   └── data_manager_bridge.py          # Sequential in-process data bridge
└── runtime_settings_configs/robots/igris_d/
    ├── __init__.py
    ├── init_params.py                  # INIT_JOINT_LIST, INIT_HAND_LIST, IGRIS_D_STATE_KEYS
    ├── inference_runtime_params.py     # RuntimeParams class
    ├── inference_runtime_params.json   # the actual values
    └── inference_runtime_topics.json   # ROS topic configuration
```

That's 5 bridges + 1 runtime config dir + 4 JSON/Python files.

### Files to edit

You also have to add `elif robot == "igris_d":` import branches in the matching interfaces:

| Interface file | Branch to add |
|---|---|
| [env_actor/robot_io_interface/controller_interface.py](../env_actor/robot_io_interface/controller_interface.py) | `from .robots.igris_d.controller_bridge import ControllerBridge` |
| [env_actor/nom_stats_manager/data_normalization_interface.py](../env_actor/nom_stats_manager/data_normalization_interface.py) | `from .robots.igris_d.data_normalization_manager import DataNormalizationBridge` |
| [env_actor/episode_recorder/episode_recorder_interface.py](../env_actor/episode_recorder/episode_recorder_interface.py) | `from env_actor.episode_recorder.robots.igris_d.episode_recorder_bridge import EpisodeRecorderBridge` |
| [env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py) | `from env_actor.auto.inference_algorithms.rtc.data_manager.robots.igris_d.shm_manager_bridge import SharedMemoryManager` |
| [env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py](../env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py) | `from env_actor.auto.inference_algorithms.sequential.data_manager.robots.igris_d.data_manager_bridge import DataManagerBridge` |
| [run_online_rl.py](../run_online_rl.py) (`start_online_rl` and argparse `--robot`) | accept `"igris_d"` and import its `RuntimeParams` |
| [env_actor/auto/inference_algorithms/rtc/rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) `start()` | import the matching `RuntimeParams` |
| RTC `control_loop.py` and `inference_loop.py` | same `RuntimeParams` import branch |

### Use `igris_b` as the template

[env_actor/robot_io_interface/robots/igris_b/](../env_actor/robot_io_interface/robots/igris_b/) is the only complete reference. Match the method signatures of every bridge — the interfaces call them by name without type-checking.

### `igris_c` — what is already there, what is missing

The repo has partial scaffolding for `igris_c`:

| Present | Missing |
|---|---|
| `controller_bridge.py` (raises NotImplementedError) | Working ROS2 communication |
| `data_manager_bridge.py` for RTC and Sequential (raises NotImplementedError) | Same |
| `episode_recorder_bridge.py` (empty file) | Everything |
| `init_params.py` (TODO placeholders) | Real joint constants and state keys |
| `__init__.py` for the runtime-config dir | **`inference_runtime_params.py`** ← without this, `--robot igris_c` crashes at import |
| — | **`inference_runtime_params.json`** |
| — | **`inference_runtime_topics.json`** |
| — | The `igris_c` branch in `nom_stats_manager/data_normalization_interface.py` |

`igris_c` is a useful template if you want to see what a half-finished port looks like. Don't try to run it.

---

## Recipe 3: Add a new inference algorithm

You want a third option alongside RTC and Sequential — say a `chunk_streaming` algorithm.

### Files to add

```text
env_actor/auto/inference_algorithms/chunk_streaming/
├── __init__.py
├── chunk_streaming_actor.py           # the Ray actor class
└── data_manager/                       # optional, mirror RTC or Sequential
    ├── data_manager_interface.py
    └── robots/igris_b/
        ├── __init__.py
        └── data_manager_bridge.py
```

### What the actor must do

Your Ray actor's `start()` method is what `run_online_rl.py` calls. The actor must:

1. Build the policy with `build_policy(policy_yaml_path)` and `.to(device)`. Call `policy.warmup()` and `policy.eval()`.
2. Either (a) accept `policy_state_manager_handle` as a constructor argument (Sequential pattern) or (b) `ray.get_actor("policy_state_manager")` from inside the loop (RTC pattern). Pick (a) — it survives renames.
3. Use a `ControllerInterface` to talk to the robot.
4. Use an `EpisodeRecorderInterface` to accumulate episodes and call `serve_train_data_buffer()` at episode end.
5. Push episodes through `episode_queue_handle.put(ray.put(td), block=True)`.
6. Use `DataNormalizationInterface` if your policy expects normalization to happen outside it (it shouldn't — see [08_invariants.md](08_invariants.md#normalization-is-inside-the-policy)).

### Files to edit

- [run_online_rl.py](../run_online_rl.py): add `chunk_streaming` to the argparse `choices=[...]`. Add an `elif inference_algorithm == 'chunk_streaming':` branch in `start_online_rl()` that imports and constructs your actor with the right `.options(...)` (at minimum `resources={"inference_pc": 1}`, plus `num_gpus=1` if you need GPU).

### Pattern to copy

[env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py) is the simplest reference — one class, one `start()` method, no subprocess shenanigans. Start from it and add complexity only as your algorithm requires.

---

## Recipe 4: Add a new reward labeler

You want a different reward signal than Robometer's progress score — say, a discriminator trained on demonstrations.

### Files to add

```text
data_labeler/<your_branch>/
├── __init__.py
└── <your_labeler>.py     # @ray.remote class with start() and process_episode()
```

### What the actor must look like

The Ray Queue and replay buffer contract is the only thing that matters. Your actor:

```python
@ray.remote(num_gpus=1)  # if you need a GPU
class MyRewardLabelerActor:
    def __init__(self, episode_queue_handle, replay_buffer_actor,
                 img_frame_key: str, reward_key: str, **other_kwargs):
        self.episode_queue_handle = episode_queue_handle
        self.replay_buffer_actor = replay_buffer_actor
        # ... build/load your scoring model

    def start(self):
        while True:
            episode_data = self.episode_queue_handle.get(block=True)
            # episode_data is the TensorDict that EpisodeRecorderBridge produced.
            #   keys include: "head", "left", "right" (T,3,H,W), "proprio" (T,D),
            #   "action" (T,A), "reward" (T,) zero-initialized, "control_mode",
            #   "episode", "task_index", ("next","done").
            # Set episode_data[self.reward_key] to your per-frame reward.
            ray.get(self.replay_buffer_actor.add.remote(episode_data))
```

The auto labeler ([data_labeler/auto/auto_reward_labeler.py](../data_labeler/auto/auto_reward_labeler.py)) is the canonical reference. The manual labeler ([data_labeler/human_in_the_loop/hil_reward_labeler.py](../data_labeler/human_in_the_loop/hil_reward_labeler.py)) shows how a Qt event loop fits inside the same protocol.

### Files to edit

- [run_online_rl.py](../run_online_rl.py): in the branch where you want to use this labeler, replace the `from data_labeler.auto.auto_reward_labeler import AutoRewardLabelerActor` import with your class, and pass the kwargs your `__init__` needs. Keep `episode_queue_handle` and `replay_buffer_actor` — those are framework-level.

### Don't forget `.start.remote()`

The auto-labeler branch in [run_online_rl.py](../run_online_rl.py) does:

```python
labeler = RewardLabeler.options(...).remote(...)
labeler.start.remote()      # ← this line
```

The manual branch as committed is **missing** the `labeler.start.remote()` call, which is why the manual labeler currently never pulls from the queue. If you copy from the manual branch, copy the missing line too.

---

Next: [08_invariants.md](08_invariants.md) — the architectural rules that all four recipes must respect.
