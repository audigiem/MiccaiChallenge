#!/bin/bash
#SBATCH --job-name=airogs_v4
#SBATCH --output=train_run/output/airogs_v4_%j.out
#SBATCH --error=train_run/stderr/airogs_v4_%j.err
#SBATCH --partition=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00

# ============================================================================
# V4 ADVANCED PREPROCESSING TRAINING SUBMISSION
# ============================================================================
# Features:
#   - Optic disk detection & cropping
#   - Advanced CLAHE (LAB color space)
#   - Vessel enhancement (optional)
#   - Weighted BCE (proven stable)
#   - Moderate augmentation
# ============================================================================

echo "=========================================="
echo "AIROGS V4 TRAINING - ADVANCED PREPROCESSING"
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

# Set environment variables for GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TF_CPP_MIN_LOG_LEVEL=0

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

# V4 preprocessing info
echo "✨ V4 ADVANCED FEATURES:"
echo "   🔬 Optic Disk Detection: ENABLED"
echo "   🎨 Advanced CLAHE: LAB color space"
echo "   🌳 Vessel Enhancement: OPTIONAL"
echo "   📊 Weighted BCE: Class weights {0: 1.0, 1: 5.0}"
echo ""

# Run training
echo "=========================================="
echo "🚀 Starting V4 Training..."
echo "=========================================="
echo ""

python3 v4_advanced/train_v4.py

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ V4 TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ V4 TRAINING FAILED (exit code: $exit_code)"
fi
echo "=========================================="
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Print output locations
echo "📁 Output locations:"
echo "   Models: v4_advanced/models/"
echo "   Logs: v4_advanced/logs/"
echo "   Checkpoints: v4_advanced/checkpoints/"
echo "   Preprocessing samples: v4_advanced/preprocessing_samples/"
echo ""

exit $exit_code
