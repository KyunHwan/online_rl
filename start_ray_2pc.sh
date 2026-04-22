#!/bin/bash
# start_ray_2pc.sh — run on each machine for the 2-PC topology
# (inference_pc + labeling_pc combined on one machine; training_pc on robros-ai1)

# Pick which machine hosts the combined inference+labeling role.
# Defaults to robros-5090 (strong GPU). Override via env var if needed.
COMBINED_HOST="${COMBINED_HOST:-robros-5090}"

# Head = the combined machine. On the trainer box, set HEAD_IP to the combined
# machine's Tailscale IP (printed by the head node on startup).
HEAD_IP="${HEAD_IP:-}"
HOSTNAME=$(hostname)

TS_IP=$(tailscale ip -4)
if [ -z "$TS_IP" ]; then
    echo "Error: Could not find a Tailscale IP. Is Tailscale running?"
    exit 1
fi

if [ "$HOSTNAME" = "$COMBINED_HOST" ]; then
    ray start --head --port=6379 \
        --node-ip-address=$TS_IP \
        --resources='{"inference_pc": 1, "labeling_pc": 1}'
    echo "Head started at $TS_IP. Set HEAD_IP=$TS_IP on the trainer box."
elif [ "$HOSTNAME" = "robros-ai1" ]; then
    if [ -z "$HEAD_IP" ]; then
        echo "Error: HEAD_IP must be set to the combined machine's Tailscale IP."
        echo "Example: HEAD_IP=100.x.y.z ./start_ray_2pc.sh"
        exit 1
    fi
    ulimit -n 65535
    ray start --address=${HEAD_IP}:6379 \
        --node-ip-address=$TS_IP \
        --resources='{"training_pc": 3}'
else
    echo "Unknown host: $HOSTNAME (expected $COMBINED_HOST or robros-ai1)"
    exit 1
fi
