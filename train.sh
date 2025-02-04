#!/bin/bash

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

# Create necessary directories
mkdir -p weights
mkdir -p dataset/split
mkdir -p logs

# Set number of GPUs
NUM_GPUS=1

# Default hyperparameters
BATCH_SIZE=32
GRAD_ACCUM_STEPS=2
EPOCHS=5
LEARNING_RATE=2e-5
MODEL_NAME="xlm-roberta-large"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum_steps)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print training configuration
echo "Training Configuration:"
echo "----------------------"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Model: $MODEL_NAME"
echo "----------------------"

# Set CUDA device order
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Create the training command
TRAIN_CMD="python model/train.py \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --model_name $MODEL_NAME \
    --fp16"

# Run the training command with nohup
echo "Starting training in background..."
nohup $TRAIN_CMD > logs/training.log 2>&1 &

# Get the process ID
TRAIN_PID=$!
echo $TRAIN_PID > logs/train.pid

echo ""
echo "Training has been started in the background (PID: $TRAIN_PID)"
echo "You can disconnect from SSH and the training will continue"
echo ""
echo "To monitor the training:"
echo "1. View logs: tail -f logs/training.log"
echo "2. Monitor on W&B: Check the URL printed in the log file"
echo "3. Check if running: ps -p $TRAIN_PID"
echo "4. Kill training: kill $TRAIN_PID"
echo ""
echo "The model will be saved in the weights directory when validation AUC improves"
echo "Training metrics and artifacts will be logged to W&B" 