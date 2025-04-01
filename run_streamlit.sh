#!/bin/bash

# Streamlit Launcher Script for Toxic Comment Classifier
# This script launches the Streamlit version of the application

echo "🚀 Starting Toxic Comment Classifier - Streamlit Edition"
echo "📚 Loading model and dependencies..."

# Check for Python and Streamlit
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 to run this application."
    exit 1
fi

if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️ Streamlit not found. Attempting to install dependencies..."
    pip install -r requirements.txt
fi

# Set default environment variables if not already set
export ONNX_MODEL_PATH=${ONNX_MODEL_PATH:-"weights/toxic_classifier.onnx"}
export PYTORCH_MODEL_DIR=${PYTORCH_MODEL_DIR:-"weights/toxic_classifier_xlm-roberta-large"}

# Run the Streamlit app
echo "✅ Launching Streamlit application..."
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 "$@" 