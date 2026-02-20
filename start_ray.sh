#!/bin/bash
# start_ray.sh — run on each machine

# Head node's Tailscale IP
HEAD_IP="100.109.184.39"
HOSTNAME=$(hostname)

# Dynamically grab this machine's Tailscale IPv4 address
TS_IP=$(tailscale ip -4)

# Ensure Tailscale is actually running/available
if [ -z "$TS_IP" ]; then
    echo "Error: Could not find a Tailscale IP. Is Tailscale running?"
    exit 1
fi

case "$HOSTNAME" in
  robros-ai1)
    ulimit -n 65535

    ray start --head --port=6379 \
      --node-ip-address=$TS_IP \
      --resources='{"training_pc": 3}'
    ;;
  robros-MS-7E59)
    ray start --address=${HEAD_IP}:6379 \
      --node-ip-address=$TS_IP \
      --resources='{"labeling_pc": 1}'
    ;;
  robros-5090)
    ray start --address=${HEAD_IP}:6379 \
      --node-ip-address=$TS_IP \
      --resources='{"inference_pc": 1}'
    ;;
  *)
    echo "Unknown host: $HOSTNAME"
    exit 1
    ;;
esac

# HEAD_IP="192.168.0.134"
# HOSTNAME=$(hostname)

# case "$HOSTNAME" in
#   robros-ai1)
#     ray start --head --port=6379 \
#       --resources='{"training_pc": 3}'
#     ;;
#   robros-MS-7E59)
#     ray start --address=${HEAD_IP}:6379 \
#       --resources='{"labeling_pc": 1}'
#     ;;
#   robros-5090)
#     ray start --address=${HEAD_IP}:6379 \
#       --resources='{"inference_pc": 1}'
#     ;;
#   *)
#     echo "Unknown host: $HOSTNAME"
#     exit 1
#     ;;
# esac