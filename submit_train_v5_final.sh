#!/bin/bash
#SBATCH --job-name=airogs_v5_final
#SBATCH --output=train_run/output/airogs_v5_final_%j.out
#SBATCH --error=train_run/stderr/airogs_v5_final_%j.err
#SBATCH --partition=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00

# ============================================================================
# V5 FINAL OPTIMIZED TRAINING
# ============================================================================
# Improvements from V4 analysis:
#   - Increased class weight: 10.0 (was 5.0)
#   - Higher initial LR: 1e-4 (was 5e-5)
#   - Better early stopping patience: 7 (was 5)
#   - Monitor val_auc (was val_loss)
# ============================================================================

echo "=========================================="
echo "AIROGS V5 FINAL - OPTIMIZED TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Print resource allocation
echo "📊 Resource Allocation:"
echo "   CPUs: $SLURM_CPUS_PER_TASK"
echo "   Memory: 32GB"
echo "   GPU: RTX 6000"
echo "   Time limit: 30 hours"
echo ""

# Environment setup
echo "🔧 Setting up environment..."
if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "No CUDA module, using system CUDA"
    module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module, using system cuDNN"
    echo "Loaded modules:"
    module list
else
    echo "Module command not available, using system CUDA/cuDNN"
fi
echo ""

echo "   Python: $(which python3)"
echo "   Python version: $(python3 --version)"
echo ""

# Check GPU
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Activate environment
echo "Activating virtual environment..."
source ~/MiccaiChallenge/bin/activate
export OMP_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=1

# Verify CUDA
echo "🔍 CUDA Environment:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu --format=csv
echo ""

# Navigate to workspace
cd /user/8/audigiem/FIB/DLMA/MiccaiChallenge/MiccaiChallenge
echo "📂 Working directory: $(pwd)"
echo ""

# V5 improvements info
echo "🎯 V5 FINAL IMPROVEMENTS:"
echo "   ⚡ Class weight increased: 1:10 (was 1:5 in V4)"
echo "   ⚡ Learning rate increased: 1e-4 (was 5e-5 in V4)"
echo "   ⚡ Early stopping patience: 7 epochs (was 5 in V4)"
echo "   ⚡ Monitoring: val_auc (was val_loss in V4)"
echo ""
echo "✅ KEPT FROM V4 (proven to work):"
echo "   - Optic disk detection & cropping"
echo "   - Advanced CLAHE (LAB color space)"
echo "   - Moderate augmentation"
echo "   - Weighted BCE loss"
echo ""

# Run training
echo "=========================================="
echo "🚀 Starting V5 Final Training..."
echo "=========================================="
echo ""

python3 v4_advanced/train_v5_final.py

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ V5 FINAL TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ V5 FINAL TRAINING FAILED (exit code: $exit_code)"
fi
echo "=========================================="
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Print output locations
echo "📁 Output locations:"
echo "   Models: v4_advanced/models_v5/"
echo "   Logs: v4_advanced/logs_v5/"
echo "   Checkpoints: v4_advanced/checkpoints_v5/"
echo "   Preprocessing samples: v4_advanced/preprocessing_samples/"
echo ""

exit $exit_code
