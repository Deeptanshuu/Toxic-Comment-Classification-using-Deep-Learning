#!/bin/bash

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

# Create necessary directories
mkdir -p weights
mkdir -p dataset/split

# Set number of GPUs
NUM_GPUS=2

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
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Model: $MODEL_NAME"
echo "----------------------"

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    model/train.py \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --model_name $MODEL_NAME \
    --fp16  # Enable mixed precision training 