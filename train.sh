#!/bin/bash

# Basic configuration
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:${PWD}"  # Add current directory to Python path

# Create directories
mkdir -p logs weights cache

# Get timestamp for error log only
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ERROR_LOG="logs/error_${TIMESTAMP}.log"

# Print configuration
echo "Starting training with configuration:"
echo "======================================"
echo "Error log: $ERROR_LOG"
echo "PYTHONPATH: $PYTHONPATH"
echo "======================================"

# Start training with nohup, only redirecting stderr
echo "Starting training in background..."
nohup python model/train.py 2> "$ERROR_LOG" &

# Save process ID
pid=$!
echo $pid > "logs/train_${TIMESTAMP}.pid"
echo "Training process started with PID: $pid"
echo
echo "Monitor commands:"
echo "1. View training progress:  tail -f logs/train_*.log"
echo "2. View error log:         tail -f $ERROR_LOG"
echo "3. Check process status:   ps -p $pid"
echo "4. Stop training:          kill $pid" 