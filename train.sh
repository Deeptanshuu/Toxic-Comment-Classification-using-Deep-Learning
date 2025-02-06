#!/bin/bash

# Setup error handling
set -e  # Exit on error
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

# Configuration
export CUDA_VISIBLE_DEVICES="0,1"  # Specify GPUs to use
export NCCL_DEBUG=INFO  # Enable NCCL debugging
export NCCL_SOCKET_IFNAME=^lo,docker0  # Avoid docker interfaces
export TORCH_DISTRIBUTED_DEBUG=INFO  # Enable PyTorch distributed debugging

# Create necessary directories
mkdir -p logs
mkdir -p weights
mkdir -p tokenized

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

# Training configuration
BATCH_SIZE=32  # Per-GPU batch size (64 effective with 2 GPUs)
GRAD_ACCUM=2   # Gradient accumulation steps
NUM_EPOCHS=10
LEARNING_RATE=1.4e-5  # Base learning rate
NUM_WORKERS=4  # Per-GPU workers
MIXED_PRECISION="bf16"  # Use bfloat16 for better stability

echo "Starting training with configuration:"
echo "======================================"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * $(nvidia-smi -L | wc -l)))"
echo "Learning rate: $LEARNING_RATE"
echo "Mixed precision: $MIXED_PRECISION"
echo "Number of workers per GPU: $NUM_WORKERS"
echo "======================================"

# Function to check GPU availability
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    if [ "$NUM_GPUS" -lt 2 ]; then
        echo "Warning: Less than 2 GPUs available. Falling back to single GPU training."
    fi
    
    # Check GPU memory
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$FREE_MEM" -lt 20000 ]; then
        echo "Warning: Less than 20GB total free GPU memory. Training might be unstable."
    fi
}

# Function to clean up on exit
cleanup() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Training failed with exit code $EXIT_CODE"
        echo "Check error log at: $ERROR_LOG"
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" &> /dev/null || true
    
    exit $EXIT_CODE
}
trap cleanup EXIT

# Check GPU availability
check_gpus

# Start training
echo "Starting training... Log file: $LOG_FILE"
python model/train.py \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --mixed_precision $MIXED_PRECISION \
    --num_workers $NUM_WORKERS \
    --model_name "xlm-roberta-large" \
    --activation_checkpointing true \
    --tensor_float_32 true \
    --gc_frequency 100 2>&1 | tee -a "$LOG_FILE" || {
        echo "Training failed. Check error log for details."
        exit 1
    }

echo "Training completed successfully!" 