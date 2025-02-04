#!/bin/bash

# Create necessary directories
mkdir -p weights
mkdir -p logs
mkdir -p .cuda_cache

# Activate virtual environment
source venv/bin/activate

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Enable CUDA optimizations
export CUDA_AUTO_TUNE=1
export CUDA_CACHE_PATH=.cuda_cache

# Set PyTorch configurations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_DISTRIBUTED_DEBUG=INFO  # More detailed DDP debugging

# Set NCCL configurations for DDP
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Get timestamp for unique log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
ERROR_LOG="logs/error_${TIMESTAMP}.log"

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Run training in background with nohup
echo "Starting training with ${NUM_GPUS} GPUs..."
nohup torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    model/train.py \
    --batch_size 64 \
    --grad_accum_steps 2 \
    --mixed_precision bf16 \
    --num_workers 12 \
    --activation_checkpointing true \
    --tensor_float_32 true \
    --gc_frequency 500 > "${LOG_FILE}" 2> "${ERROR_LOG}" &

# Save the process ID
echo $! > logs/train.pid

echo ""
echo "Training has been started in the background with PID: $(cat logs/train.pid)"
echo "Using ${NUM_GPUS} GPUs for distributed training"
echo ""
echo "To monitor the training:"
echo "1. View logs: tail -f ${LOG_FILE}"
echo "2. View errors: tail -f ${ERROR_LOG}"
echo "3. Monitor GPU: nvidia-smi -l 1"
echo "4. Check if running: ps -p $(cat logs/train.pid)"
echo "5. Kill training if needed: kill $(cat logs/train.pid)"
echo ""
echo "The model will be saved in weights/ directory"
echo "Training metrics are logged to W&B and ${LOG_FILE}"

# Monitor for immediate startup errors
sleep 5
if ! ps -p $(cat logs/train.pid) > /dev/null; then
    echo "Error: Training process failed to start. Check error log:"
    cat "${ERROR_LOG}"
    exit 1
fi 