#!/bin/bash

# Setup error handling
set -e  # Exit on error
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

# Configuration
export CUDA_VISIBLE_DEVICES="0,1"  # Specify GPUs to use
export NCCL_DEBUG=WARN  # Reduce NCCL logging
export NCCL_SOCKET_IFNAME=^lo,docker0  # Avoid docker interfaces
export TORCH_DISTRIBUTED_DEBUG=OFF  # Disable distributed debugging for performance
export CUDA_LAUNCH_BLOCKING=1  # Better error reporting
export TORCH_USE_CUDA_DSA=1  # Enable CUDA Graph memory optimizations
export OMP_NUM_THREADS=1  # Prevent numpy/OpenMP thread contention

# Suppress CUDA registration warnings
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export TF_CPP_MIN_LOG_LEVEL=2 # To suppress tensorflow warnings
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_MODULE_LOADING="LAZY"

# Create necessary directories
mkdir -p logs
mkdir -p weights
mkdir -p tokenized
mkdir -p cache

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

# Training configuration
BATCH_SIZE=24  # Reduced per-GPU batch size for stability
GRAD_ACCUM=2   # Gradient accumulation steps
NUM_EPOCHS=10
LEARNING_RATE=1.4e-5  # Base learning rate
NUM_WORKERS=2  # Reduced workers per GPU
MIXED_PRECISION="bf16"  # Use bfloat16 for better stability

# Function to check GPU availability and memory
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    if [ "$NUM_GPUS" -lt 2 ]; then
        echo "Warning: Less than 2 GPUs available. Falling back to single GPU training."
        BATCH_SIZE=$((BATCH_SIZE * 2))  # Double batch size for single GPU
        echo "Adjusted batch size to: $BATCH_SIZE"
    fi
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" &> /dev/null || true
    
    # Check GPU memory
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$FREE_MEM" -lt 20000 ]; then
        echo "Warning: Less than 20GB total free GPU memory. Training might be unstable."
        BATCH_SIZE=$((BATCH_SIZE / 2))  # Reduce batch size
        GRAD_ACCUM=$((GRAD_ACCUM * 2))  # Increase accumulation to compensate
        echo "Adjusted configuration for low memory:"
        echo "Batch size: $BATCH_SIZE"
        echo "Gradient accumulation: $GRAD_ACCUM"
    fi
}

# Function to clean up on exit
cleanup() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Training failed with exit code $EXIT_CODE"
        echo "Check error log at: $ERROR_LOG"
        
        # Print GPU state
        echo "GPU state at failure:"
        nvidia-smi
        
        # Print last few lines of error log
        echo "Last few lines of error log:"
        tail -n 20 "$ERROR_LOG"
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" &> /dev/null || true
    
    exit $EXIT_CODE
}
trap cleanup EXIT

# Check GPU availability and adjust parameters
check_gpus

echo "Starting training with configuration:"
echo "======================================"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * $(nvidia-smi -L | wc -l)))"
echo "Learning rate: $LEARNING_RATE"
echo "Mixed precision: $MIXED_PRECISION"
echo "Number of workers per GPU: $NUM_WORKERS"
echo "Activation checkpointing: true"
echo "Tensor float 32: true"
echo "======================================"

# Start training
echo "Starting training... Log file: $LOG_FILE"
nohup python -u model/train.py \
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