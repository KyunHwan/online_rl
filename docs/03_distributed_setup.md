# 03 — Distributed setup

This page explains how the three machines find each other, how Ray's [custom resources](10_glossary.md#custom-resource) pin actors to specific machines, and what a new operator must edit to redeploy the system elsewhere.

## Table of contents

- [Why Tailscale](#why-tailscale)
- [Hostname-based routing](#hostname-based-routing)
- [The current topology](#the-current-topology)
- [How resources pin actors to machines](#how-resources-pin-actors-to-machines)
- [The `--node-ip-address` flag](#the---node-ip-address-flag)
- [Single-machine development](#single-machine-development)
- [Deploying on a LAN without Tailscale](#deploying-on-a-lan-without-tailscale)
- [What to change for a new deployment](#what-to-change-for-a-new-deployment)

## Why Tailscale

[Tailscale](10_glossary.md#tailscale) gives every machine a stable IPv4 address in the `100.x.y.z` range, independent of the underlying physical or VPN network. The three machines in this deployment can sit on different subnets, behind different NATs, or move between offices and still see each other on a flat, encrypted overlay.

Ray needs every node to have a routable address for inter-node RPC. If the head node says "connect to me at 192.168.1.10" but the inference machine is on a different subnet, peering breaks. Tailscale sidesteps that — the head's `100.x.y.z` address works from anywhere on the tailnet.

## Hostname-based routing

[start_ray.sh](../start_ray.sh) is one script shared by all three machines. It uses `$HOSTNAME` to figure out which branch to take:

```bash
HEAD_IP="100.118.28.46"          # Head node's Tailscale IPv4
TS_IP=$(tailscale ip -4)         # This machine's Tailscale IPv4

case "$HOSTNAME" in
  robros-MS-7E59)                # The labeling/head machine
    ray start --head --port=6379 \
      --node-ip-address=$TS_IP \
      --resources='{"labeling_pc": 4}'
    ;;
  robros-ai1)                    # The training machine
    ulimit -n 65535
    ray start --address=${HEAD_IP}:6379 \
      --node-ip-address=$TS_IP \
      --resources='{"training_pc": 3}'
    ;;
  robros-5090)                   # The inference machine
    ray start --address=${HEAD_IP}:6379 \
      --node-ip-address=$TS_IP \
      --resources='{"inference_pc": 1}'
    ;;
esac
```

If `$HOSTNAME` matches none of these (`*` branch), the script exits with `Unknown host: $HOSTNAME`.

This means **the hostnames and the head IP are deployment-specific values, not framework defaults**. If you redeploy on three different machines, you edit `start_ray.sh` first.

## The current topology

| Hostname | Role | Resource label | `ray start` mode |
|---|---|---|---|
| `robros-MS-7E59` | Labeling + cluster head | `labeling_pc: 4` | `--head` |
| `robros-ai1` | Training | `training_pc: 3` | `--address=$HEAD_IP:6379` |
| `robros-5090` | Inference (robot-side) | `inference_pc: 1` | `--address=$HEAD_IP:6379` |

The labeling machine is also the head. The top-level [README.md](../README.md) historically said the head was the training machine — verify against [start_ray.sh](../start_ray.sh), which is authoritative: the `--head` flag is on the `robros-MS-7E59` branch.

`ulimit -n 65535` is only set on `robros-ai1`. The trainer runs Ray Train DDP with 4 workers, each holding many memmap files and dataloader fds open — the default 1024 limit is too low. If your training machine has a different hostname, add the `ulimit` line to its branch.

## How resources pin actors to machines

Ray's `--resources='{"<label>": N}'` declares that a node has `N` units of a custom resource. Actors declared with `.options(resources={"<label>": K})` will only be scheduled on a node that has at least `K` units free.

[run_online_rl.py](../run_online_rl.py) pins actors this way:

| Actor | `.options(...)` | Lands on |
|---|---|---|
| `StateManagerActor` | `resources={"training_pc": 1}` | `robros-ai1` |
| `ReplayBufferActor` | `resources={"training_pc": 1}` | `robros-ai1` |
| `RTCActor` / `SequentialActor` | `resources={"inference_pc": 1}` (+ `num_gpus=1`) | `robros-5090` |
| `run_training` (Ray task) | `resources={"training_pc": 1}` | `robros-ai1`; spawns 4 DDP workers inside that resource |
| `AutoRewardLabelerActor` | `resources={"labeling_pc": 1}` per actor, `num_gpus=1` | `robros-MS-7E59`; 4 of them with `--num_labeler_gpus=4` |
| `ManualRewardLabelerActor` | `resources={"labeling_pc": 1}` | `robros-MS-7E59` |

Counts in `start_ray.sh` set the upper bound. `labeling_pc: 4` allows up to 4 auto-labeler actors per labeling node. `training_pc: 3` is generous (only one trainer + one replay buffer + one state manager land there; 3 is more than enough). `inference_pc: 1` allows exactly one RTC/Sequential actor.

If you try to launch more actors than the available resource units, Ray will queue them indefinitely (no error, just nothing happens). Run `ray status` to verify capacities.

## The `--node-ip-address` flag

Each machine passes its own Tailscale address with `--node-ip-address=$TS_IP`. Without this, Ray might guess the machine's primary IP from `/etc/hosts` or DNS, which could resolve to an unreachable LAN address. Forcing the Tailscale address ensures inter-node connections use the tailnet.

`HEAD_IP` is the **head's** Tailscale address. Worker nodes need this constant to find the head. The head itself uses its own `$TS_IP` so it advertises the same address it expects clients to connect to.

## Single-machine development

If you only have one machine and you want to test the full pipeline locally, the cleanest workaround is to make `start_ray.sh` start a single head with all three resource labels at once. The simplest edit is:

```bash
# Replace the case statement with a single ray start --head
ray start --head --port=6379 \
  --node-ip-address=127.0.0.1 \
  --resources='{"labeling_pc": 4, "training_pc": 3, "inference_pc": 1}'
```

Every actor will then land on that one machine. GPU memory becomes the bottleneck — you cannot run 4 auto labelers + 1 inference actor + 4 trainer DDP workers on one consumer GPU. Reduce `--num_labeler_gpus` to 1 (or use `--human_reward_labeler` to skip the auto labelers entirely) and the trainer `ScalingConfig(num_workers=1)` in [run_online_rl.py](../run_online_rl.py) line 40 if you actually need to run it on one GPU.

## Deploying on a LAN without Tailscale

The bottom of [start_ray.sh](../start_ray.sh) has a commented-out variant for LAN deployments:

```bash
# HEAD_IP="192.168.0.134"
# case "$HOSTNAME" in
#   robros-ai1)
#     ray start --head --port=6379 --resources='{"training_pc": 3}'
#     ;;
#   ...
# esac
```

That alternate block does not pass `--node-ip-address` and uses a LAN IP for the head. Use it when all three machines are on the same subnet and you do not need Tailscale's NAT traversal.

To switch: comment out the active block, uncomment the LAN block, edit `HEAD_IP` to your training machine's LAN address, and adjust the hostnames to your machines.

## What to change for a new deployment

For a junior reader redeploying this system on three new machines, the minimal edit list:

1. **[start_ray.sh](../start_ray.sh)** — update `HEAD_IP`, the three hostnames in the `case` block, and possibly the resource counts.
2. **[env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json)** — update `norm_stats_file_path` to point at your local copy of `dataset_stats.pkl`.
3. **[run_online_rl.py](../run_online_rl.py)** — if you do not pass `--train_config` explicitly, change the default path on line 179 to your repo location. (Editing code is out of scope for these docs but the path is hard-coded to `/home/user/Projects/online_rl/...`.)

Code-level resource pinning (the `resources={"..."}` calls in [run_online_rl.py](../run_online_rl.py) and in actors' `.options(...)`) does not need to change — as long as you keep the same three resource labels (`labeling_pc`, `training_pc`, `inference_pc`), actors will land on whichever machines you assign those labels to.

Next: [04_configuration.md](04_configuration.md) explains the three config layers that control what gets built and what shapes the data takes.
