#!/bin/bash

# Basic configuration
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:${PWD}"  # Add current directory to Python path


# Create directories
mkdir -p logs weights cache

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"
ERROR_LOG="logs/error_${TIMESTAMP}.log"

# Print
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo "PYTHONPATH: $PYTHONPATH"
echo "======================================"

# Start training with nohup
echo "Starting training in background..."
nohup python model/train.py > "$LOG_FILE" 2> "$ERROR_LOG" &

# Save process ID
pid=$!
echo $pid > "logs/train_${TIMESTAMP}.pid"
echo "Training process started with PID: $pid"
echo
echo "Monitor commands:"
echo "3. Check process status:   ps -p $pid"
echo "4. Stop training:          kill $pid" 