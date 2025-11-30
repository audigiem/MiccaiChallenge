#!/bin/bash
#SBATCH --job-name=airogs_quick
#SBATCH --output=logs/airogs_quick_%j.out
#SBATCH --error=logs/airogs_quick_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Quick training script for testing (fewer epochs, smaller batch)
# Usage: sbatch train_cluster_quick.sh

echo "========================================"
echo "AIROGS Quick Test Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Load required modules
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment
source .venv/bin/activate

# Create directories
mkdir -p logs outputs/models outputs/plots

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

# Run quick training (5 epochs for testing)
echo "Starting quick training (5 epochs)..."
python train.py \
    --data-dir ./data \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --image-size 384 \
    --backbone efficientnet-b0

echo ""
echo "========================================"
echo "Quick training completed!"
echo "End time: $(date)"
echo "========================================"

deactivate

