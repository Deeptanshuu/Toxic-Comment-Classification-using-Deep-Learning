#!/bin/bash

# Initialize variables for trap
current_command=""
last_command=""

# Strict mode with allowance for unbound variables in traps
set -eo pipefail  # Removed 'u' flag to allow unbound vars in traps
IFS=$'\n\t'      # Stricter word splitting

# Function to clean up on exit
cleanup() {
    local EXIT_CODE=$?
    
    # Only show error messages if there was an error
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Training failed with exit code $EXIT_CODE"
        echo "Check error log at: $ERROR_LOG"
        
        # Print GPU state safely
        echo "GPU state at failure:"
        nvidia-smi || echo "Could not get GPU state"
        
        # Print last few lines of error log if it exists
        if [ -f "$ERROR_LOG" ]; then
            echo "Last few lines of error log:"
            tail -n 20 "$ERROR_LOG" || echo "Could not read error log"
        fi
    fi
    
    # Kill any remaining background processes
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null; then
            echo "Stopping training process..."
            kill "$pid" 2>/dev/null || true
            sleep 2  # Give process time to cleanup
            kill -9 "$pid" 2>/dev/null || true  # Force kill if still running
        fi
        rm -f "$PID_FILE"
    fi
    
    # Clear GPU cache safely
    python -c "import torch; torch.cuda.empty_cache()" &> /dev/null || true
    
    # Remove lock files if they exist
    rm -f ./*.lock 2>/dev/null || true
    
    exit $EXIT_CODE
}

# Setup error handling
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR
trap cleanup EXIT

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

# Verify Python and required packages are available
command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting." >&2; exit 1; }
python -c "import torch" >/dev/null 2>&1 || { echo "PyTorch is required but not installed. Aborting." >&2; exit 1; }

# Create necessary directories with error checking
for dir in logs weights tokenized cache; do
    if ! mkdir -p "$dir"; then
        echo "Error: Could not create directory: $dir" >&2
        exit 1
    fi
done

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

# Verify log files can be created
touch "$LOG_FILE" "$ERROR_LOG" "$PID_FILE" || { echo "Error: Cannot create log files" >&2; exit 1; }

# Training configuration with validation
BATCH_SIZE=32  # Standard batch size
GRAD_ACCUM=1   # No gradient accumulation
NUM_EPOCHS=10
LEARNING_RATE=2e-5  # Standard learning rate for transformers
NUM_WORKERS=2  # Reduced workers per GPU
MIXED_PRECISION="no"  # Disable mixed precision for stability

# Validate configuration
[[ $BATCH_SIZE -gt 0 ]] || { echo "Error: Invalid batch size" >&2; exit 1; }
[[ $GRAD_ACCUM -gt 0 ]] || { echo "Error: Invalid gradient accumulation steps" >&2; exit 1; }
[[ $NUM_EPOCHS -gt 0 ]] || { echo "Error: Invalid number of epochs" >&2; exit 1; }
[[ $NUM_WORKERS -ge 0 ]] || { echo "Error: Invalid number of workers" >&2; exit 1; }

# Function to check GPU availability and memory
check_gpus() {
    local nvidia_smi_output
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed." >&2
        exit 1
    fi
    
    # Check if GPUs are actually accessible
    if ! nvidia_smi_output=$(nvidia-smi); then
        echo "Error: Cannot access GPUs. Check nvidia-smi output:" >&2
        echo "$nvidia_smi_output" >&2
        exit 1
    fi
    
    local NUM_GPUS
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    
    if [ "$NUM_GPUS" -lt 2 ]; then
        echo "Warning: Less than 2 GPUs available. Falling back to single GPU training."
        BATCH_SIZE=$((BATCH_SIZE * 2))  # Double batch size for single GPU
        echo "Adjusted batch size to: $BATCH_SIZE"
        export CUDA_VISIBLE_DEVICES="0"  # Use only first GPU
    fi
    
    # Clear GPU cache safely
    if ! python -c "import torch; torch.cuda.empty_cache()" &> /dev/null; then
        echo "Warning: Could not clear GPU cache. Continuing anyway..."
    fi
    
    # Check GPU memory with error handling
    local FREE_MEM
    if ! FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}'); then
        echo "Warning: Could not query GPU memory. Using conservative settings."
        FREE_MEM=0
    fi
    
    if [ "$FREE_MEM" -lt 20000 ]; then
        echo "Warning: Less than 20GB total free GPU memory. Training might be unstable."
        BATCH_SIZE=$((BATCH_SIZE / 2))  # Reduce batch size
        GRAD_ACCUM=$((GRAD_ACCUM * 2))  # Increase accumulation to compensate
        echo "Adjusted configuration for low memory:"
        echo "Batch size: $BATCH_SIZE"
        echo "Gradient accumulation: $GRAD_ACCUM"
    fi
    
    # Verify CUDA is available in Python
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        echo "Error: CUDA is not available in Python environment" >&2
        exit 1
    fi
}

# Verify training script exists
if [ ! -f "model/train.py" ]; then
    echo "Error: Training script not found at model/train.py" >&2
    exit 1
fi

# Check GPU availability and adjust parameters
check_gpus

# Print configuration
echo "Starting training with configuration:"
echo "======================================"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * $(nvidia-smi -L | wc -l)))"
echo "Learning rate: $LEARNING_RATE"
echo "Mixed precision: $MIXED_PRECISION"
echo "Number of workers per GPU: $NUM_WORKERS"
echo "Activation checkpointing: true"
echo "Tensor float 32: true"
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo "PID file: $PID_FILE"
echo "======================================"

# Start training with nohup and proper error handling
echo "Starting training in background... Log file: $LOG_FILE"
if ! nohup python -u model/train.py \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --mixed_precision "$MIXED_PRECISION" \
    --num_workers "$NUM_WORKERS" \
    --model_name "xlm-roberta-large" \
    --activation_checkpointing true \
    --tensor_float_32 true \
    --gc_frequency 100 > "$LOG_FILE" 2> "$ERROR_LOG" & then
    echo "Error: Failed to start training process" >&2
    exit 1
fi

# Save process ID with verification
pid=$!
if ! ps -p $pid > /dev/null; then
    echo "Error: Training process failed to start" >&2
    exit 1
fi
echo $pid > "$PID_FILE"

echo "Training process started successfully with PID $pid"
echo
echo "Monitoring Commands:"
echo "-------------------"
echo "1. View training progress:"
echo "   tail -f $LOG_FILE"
echo
echo "2. View error log:"
echo "   tail -f $ERROR_LOG"
echo
echo "3. Check if process is running:"
echo "   ps -p $pid"
echo
echo "4. Monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo
echo "5. Stop training:"
echo "   kill $pid"
echo
echo "6. Force stop training (if unresponsive):"
echo "   kill -9 $pid" 