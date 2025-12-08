#!/bin/bash
#SBATCH --job-name=airogs_train_014
#SBATCH --output=train_run/output/airogs_train_014_%j.out
#SBATCH --error=train_run/stderr/airogs_train_014_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=v100
#SBATCH --gres=gpu:1

# Training script for datasets 0, 1, and 4 (no augmentation)
# This script trains the baseline model on multiple datasets

echo "=============================================="
echo "AIROGS Training - Datasets 0, 1, 4"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Configuration
MEMORY="32G"
CPUS="8"
TIME_LIMIT="08:00:00"
# Try multiple possible venv paths
if [ -f "$HOME/MiccaiChallenge/bin/activate" ]; then
    ENV_PATH="$HOME/MiccaiChallenge/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    ENV_PATH="venv/bin/activate"
elif [ -f "../venv/bin/activate" ]; then
    ENV_PATH="../venv/bin/activate"
else
    ENV_PATH=""
fi
SCRIPT_NAME="train.py"

echo "Configuration:"
echo "  Memory: $MEMORY"
echo "  CPUs: $CPUS"
echo "  Time limit: $TIME_LIMIT"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load CUDA modules (adjust versions if needed)
echo "Loading CUDA modules..."
if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "No CUDA module found, using system CUDA"
    module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module found, using system cuDNN"
else
    echo "Module command not available, using system CUDA/cuDNN"
fi

# Display loaded modules
echo "Loaded modules:"
if command -v module &> /dev/null; then
    module list
else
    echo "Module system not available"
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Activate environment and run training
echo "Activating virtual environment..."
if [ -n "$ENV_PATH" ] && [ -f "$ENV_PATH" ]; then
    source ${ENV_PATH}
    echo "Virtual environment activated: $ENV_PATH"
else
    echo "No virtual environment found, using system Python"
fi
export OMP_NUM_THREADS=${CPUS}
export TF_CPP_MIN_LOG_LEVEL=1

# Fix cuDNN convolution algorithm picker issues
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export TF_ENABLE_ONEDNN_OPTS=0

echo ""
echo "=============================================="
echo "Starting training on datasets 0, 1, 4"
echo "=============================================="
echo "Configuration:"
echo "  - No data augmentation"
echo "  - Class weights: 1.0 (NRG) / 10.0 (RG)"
echo "  - Datasets: dataset/0, dataset/1, dataset/4"
echo ""

python3 ${SCRIPT_NAME}

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"

    # List generated models
    if [ -d "outputs/models" ]; then
        echo ""
        echo "Generated models:"
        ls -lh outputs/models/*.h5 | tail -5
    fi
else
    echo "Training failed!"
fi

echo "=============================================="

