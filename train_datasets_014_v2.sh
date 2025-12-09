#!/bin/bash
#SBATCH --job-name=airogs_train_v2
#SBATCH --output=train_run/output/airogs_train_v2_%j.out
#SBATCH --error=train_run/stderr/airogs_train_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --nodes=1

# ============================================
# AIROGS Training Script - Version 2
# ============================================
# Improvements:
# - Reduced learning rate: 5e-5 (was 1e-4)
# - Reduced class weights: 5.0 (was 10.0)
# - Data augmentation ENABLED
# - Increased dropout: 0.5 (was 0.3)
# - L2 regularization added: 0.001
# - Early stopping patience: 8 (was 5)
# - Training on datasets 0, 1, 4
# ============================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directories
mkdir -p train_run/output train_run/stderr

# Job information
echo "=============================================="
echo "AIROGS Training v2 - Datasets 0, 1, 4"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Memory: ${SLURM_MEM_PER_NODE}M"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Time limit: 16:00:00"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo ""

# Load CUDA modules if available
echo "Loading CUDA modules..."
if command -v module &> /dev/null; then
    module load cuda/11.8 2>/dev/null || echo "CUDA 11.8 not available"
    module load cudnn/8.6 2>/dev/null || echo "cuDNN 8.6 not available"
    echo "Loaded modules:"
    module list
else
    echo "Module command not available, using system CUDA/cuDNN"
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "$HOME/MiccaiChallenge/bin/activate" ]; then
    source "$HOME/MiccaiChallenge/bin/activate"
    echo "Virtual environment activated: $VIRTUAL_ENV"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "Warning: Virtual environment not found, using system Python"
fi
echo ""

# Training parameters
echo "=============================================="
echo "Starting training on datasets 0, 1, 4"
echo "=============================================="
echo "Improvements over v1:"
echo "  - Data augmentation: ENABLED"
echo "  - Learning rate: 5e-5 (reduced)"
echo "  - Class weights: 1.0 (NRG) / 5.0 (RG)"
echo "  - Dropout: 0.5 (increased)"
echo "  - L2 regularization: 0.001"
echo "  - Early stopping patience: 8"
echo "  - Datasets: dataset/0, dataset/1, dataset/4"
echo ""

# Set TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=1  # Reduce TF logging
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Run training
python train.py

# Check exit status
EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training completed with exit code: $EXIT_CODE"
    echo "Training failed!"
fi
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE

