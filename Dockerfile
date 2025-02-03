# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data and models
RUN mkdir -p dataset/final_balanced weights

# Set environment variables
ENV PYTHONPATH=/app
ENV WANDB_API_KEY=""

# Default command to run training
CMD ["python", "model/train.py"] 