#!/bin/bash
#SBATCH --job-name=airogs_baseline
#SBATCH --output=logs/airogs_baseline_%j.out
#SBATCH --error=logs/airogs_baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

# AIROGS Glaucoma Detection - Baseline Training Script
# Usage: sbatch train_cluster.sh

echo "========================================"
echo "AIROGS Baseline Training - SLURM Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Load required modules (adjust based on your cluster)
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment
source .venv/bin/activate

# Print environment info
echo ""
echo "Python version:"
python --version
echo ""
echo "TensorFlow version:"
python -c "import tensorflow as tf; print(tf.__version__)"
echo ""
echo "GPU availability:"
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p outputs/models
mkdir -p outputs/plots

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "Starting training..."
python train.py \
    --data-dir ./data \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --image-size 384 \
    --backbone efficientnet-b0

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training completed successfully!"
    echo "End time: $(date)"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Training failed with exit code $?"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi

# Deactivate virtual environment
deactivate

