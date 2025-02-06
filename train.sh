#!/bin/bash

# Basic configuration
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONWARNINGS="ignore"

# Training parameters
BATCH_SIZE=32
GRAD_ACCUM=1
NUM_EPOCHS=10
LEARNING_RATE=2e-5
NUM_WORKERS=2
MIXED_PRECISION="no"

# Create directories
mkdir -p logs weights cache

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"
ERROR_LOG="logs/error_${TIMESTAMP}.log"

# Print configuration
echo "Starting training with configuration:"
echo "======================================"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Mixed precision: $MIXED_PRECISION"
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo "======================================"

# Start training with nohup
echo "Starting training in background..."
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
    --gc_frequency 100 > "$LOG_FILE" 2> "$ERROR_LOG" &

# Save process ID
pid=$!
echo $pid > "logs/train_${TIMESTAMP}.pid"
echo "Training process started with PID: $pid"
echo
echo "Monitor commands:"
echo "1. View training progress:  tail -f $LOG_FILE"
echo "2. View error log:         tail -f $ERROR_LOG"
echo "3. Check process status:   ps -p $pid"
echo "4. Stop training:          kill $pid" 