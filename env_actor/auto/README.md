# env_actor/auto

Fully autonomous policy inference — the policy runs on the robot without human intervention. This is the primary inference mode for online RL.

## Inference Algorithms

Two algorithms are available, selected via `--inference_algorithm` in [`run_online_rl.py`](../../run_online_rl.py):

### RTC (Real-Time Action Chunking)

**Default algorithm.** Designed for real-time control where GPU inference latency would otherwise cause gaps in action execution.

```
┌──────────────────────────────────────────────────────────┐
│                   RTCActor (Ray Actor)                    │
│                                                          │
│  ┌────────────────────────┐  SharedMemory  ┌──────────┐ │
│  │   Inference Process    │◄──────────────►│ Control  │ │
│  │   (GPU)                │                │ Process  │ │
│  │                        │                │ (CPU)    │ │
│  │ • build_policy()       │  proprio       │          │ │
│  │ • guided_inference()   │  cameras       │ • read   │ │
│  │ • weight updates       │  actions       │   state  │ │
│  │                        │  counters      │ • publish│ │
│  │                        │                │   action │ │
│  └────────────────────────┘                └──────────┘ │
│                                                          │
│  Synchronization: RLock + Conditions + Events            │
└──────────────────────────────────────────────────────────┘
```

**How it works:**

1. `RTCActor` (a Ray actor) creates shared memory blocks in the parent process for: proprioception history, 3 camera images (head, left, right), and the action chunk.
2. Two child processes are spawned via `multiprocessing.Process` (spawn context):
   - **Inference process** ([`actors/inference_loop.py`](inference_algorithms/rtc/actors/inference_loop.py)) — builds the policy on GPU, runs `guided_inference()` in a loop, writes new action chunks to shared memory.
   - **Control process** ([`actors/control_loop.py`](inference_algorithms/rtc/actors/control_loop.py)) — reads robot state at a fixed frequency, writes observations to shared memory, reads the current action from shared memory, and publishes it to the robot.
3. The inference process uses **guided inference** (action inpainting) to blend the previously predicted action chunk with the new one, compensating for the inference delay.
4. Delay tracking via a `MaxDeque` estimates how many control steps occur during one inference call, which feeds into the blending weights.

**Key files:**

| File | Purpose |
|------|---------|
| [`rtc/rtc_actor.py`](inference_algorithms/rtc/rtc_actor.py) | Ray actor: creates shared memory, spawns processes, manages lifecycle |
| [`rtc/actors/inference_loop.py`](inference_algorithms/rtc/actors/inference_loop.py) | GPU process: policy build, inference loop, weight updates |
| [`rtc/actors/control_loop.py`](inference_algorithms/rtc/actors/control_loop.py) | CPU process: robot I/O, episode recording, timing |
| [`rtc/data_manager/shm_manager_interface.py`](inference_algorithms/rtc/data_manager/shm_manager_interface.py) | Robot-agnostic shared memory API |
| [`rtc/data_manager/robots/igris_b/shm_manager_bridge.py`](inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) | igris_b shared memory implementation |
| [`rtc/data_manager/utils/shared_memory_utils.py`](inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py) | `create_shared_ndarray()`, `attach_shared_ndarray()`, `ShmArraySpec` |
| [`rtc/data_manager/utils/max_deque.py`](inference_algorithms/rtc/data_manager/utils/max_deque.py) | Circular buffer tracking max inference delay |

### Sequential

A simpler, single-threaded synchronous loop. No shared memory or multi-process coordination.

```
while running:
    obs = controller.read_state()        # numpy
    actions = policy.predict(obs, norm)  # numpy → [torch inside] → numpy
    controller.publish_action(actions)
```

**Key files:**

| File | Purpose |
|------|---------|
| [`sequential/sequential_actor.py`](inference_algorithms/sequential/sequential_actor.py) | Ray actor: single-threaded inference + control loop |
| [`sequential/data_manager/data_manager_interface.py`](inference_algorithms/sequential/data_manager/data_manager_interface.py) | Robot-agnostic data manager API |
| [`sequential/data_manager/robots/igris_b/data_manager_bridge.py`](inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py) | igris_b in-process observation history + action buffering |

## Shared Memory Layout (RTC)

The parent process allocates these shared memory arrays, then passes `ShmArraySpec` objects to child processes which attach by name:

| Key | Shape | Dtype | Purpose |
|-----|-------|-------|---------|
| `proprio` | `(proprio_history_size, proprio_state_dim)` | `float32` | Proprioceptive state history |
| `head` | `(num_img_obs, 3, H, W)` | `uint8` | Head camera images |
| `left` | `(num_img_obs, 3, H, W)` | `uint8` | Left camera images |
| `right` | `(num_img_obs, 3, H, W)` | `uint8` | Right camera images |
| `action` | `(action_chunk_size, action_dim)` | `float32` | Current action chunk |

Dimensions come from `RuntimeParams`, which is loaded from the robot's JSON config.

## Synchronization Primitives (RTC)

All created via `multiprocessing.get_context("spawn")`:

| Primitive | Purpose |
|-----------|---------|
| `RLock` | Mutual exclusion for shared memory access |
| `Condition(lock)` — `control_iter_cond` | Control loop notifies inference when actions have been consumed |
| `Condition(lock)` — `inference_ready_cond` | Inference notifies control when it is ready for a new episode |
| `Event` — `stop_event` | Global shutdown signal |
| `Event` — `episode_complete_event` | Signals end of an episode |
| `Value('i')` — `num_control_iters` | Counter of control steps since last inference |
| `Value(c_bool)` — `inference_ready_flag` | Whether inference process is ready |

## Episode Lifecycle (RTC)

1. **Inference process** signals ready → **Control process** receives signal.
2. Control resets robot, bootstraps observation history in shared memory.
3. Control loop runs for `episode_length` steps (default 1000):
   - Read state → write to shared memory → read action → publish.
4. When `num_control_iters` reaches `min_num_actions_executed` (default 35), inference wakes up.
5. Inference reads state snapshot, runs `guided_inference()`, writes new action chunk.
6. At episode end, control signals `episode_complete`, records episode data, pushes to Ray Queue.
7. Inference checks for updated weights from `StateManager`, then signals ready for next episode.

## Adding a New Robot

1. Create `inference_algorithms/rtc/data_manager/robots/<robot>/shm_manager_bridge.py`.
2. Create `inference_algorithms/sequential/data_manager/robots/<robot>/data_manager_bridge.py`.
3. Add the import branch in `shm_manager_interface.py` and `data_manager_interface.py`.
