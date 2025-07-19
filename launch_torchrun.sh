#!/bin/bash

# Simple script to launch DDP training with torchrun
# Usage: ./launch_torchrun.sh [num_gpus]

NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "Launching distributed training on $NUM_GPUS GPUs using torchrun"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py
