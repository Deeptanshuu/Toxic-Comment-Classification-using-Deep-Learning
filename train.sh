#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Enable CUDA optimizations
export CUDA_AUTO_TUNE=1
export CUDA_CACHE_PATH=.cuda_cache

# Set PyTorch configurations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Set NCCL configurations for DDP
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Run distributed training
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29500 \
    model/train.py \
    --batch_size 64 \
    --grad_accum_steps 2 \
    --mixed_precision bf16 \
    --num_workers 12 \
    --activation_checkpointing true \
    --tensor_float_32 true \
    --gc_frequency 500 