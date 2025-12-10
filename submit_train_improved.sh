#!/bin/bash
#
# Training script for Week 2 improvements (Focal Loss + CLAHE + Enhanced Augmentation)
# Optimized for NVIDIA A40 GPUs (ampere partition)
#

echo "============================================================"
echo "AIROGS Week 2 Improved Training - Datasets 0, 1, 4"
echo "============================================================"
echo ""

# Check datasets exist
DATASETS=("dataset/0" "dataset/1" "dataset/4")
MISSING=0

echo "🔍 Verifying datasets..."
for dataset in "${DATASETS[@]}"; do
    if [ ! -d "$dataset" ]; then
        echo "   ❌ $dataset - NOT FOUND"
        MISSING=1
    else
        COUNT=$(ls "$dataset"/*.jpg 2>/dev/null | wc -l)
        echo "   ✅ $dataset - $COUNT images"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "❌ Error: Some datasets are missing!"
    exit 1
fi

# Check labels file
echo ""
echo "🔍 Verifying labels file..."
if [ ! -f "dataset/train_labels.csv" ]; then
    echo "   ❌ dataset/train_labels.csv - NOT FOUND"
    exit 1
else
    LINES=$(wc -l < "dataset/train_labels.csv")
    echo "   ✅ dataset/train_labels.csv - $LINES lines"
fi

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p train_run/{stderr,output}
mkdir -p outputs_improved/{models,logs,checkpoints}
echo "   ✅ Directories created"

# Display configuration
echo ""
echo "📋 Week 2 Improved Configuration:"
echo "   Datasets: 0, 1, 4"
echo "   🆕 Focal Loss: ENABLED (γ=2.0)"
echo "   🆕 CLAHE Preprocessing: ENABLED"
echo "   🆕 Enhanced Augmentation: ENABLED"
echo "   Batch size: 32"
echo "   Epochs: 30"
echo "   Image size: 384x384"
echo "   GPU: 1x NVIDIA A40 (46GB VRAM)"
echo "   Partition: ampere"
echo "   Memory: 32GB"
echo "   CPUs: 8"
echo "   Time limit: 12 hours"
echo ""

# Confirm
read -p "Launch improved training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 1
fi

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="train_run/slurm_train_improved_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << 'EOFSLURM'
#!/bin/bash
#SBATCH --job-name=airogs_improved
#SBATCH --output=train_run/output/airogs_improved_%j.out
#SBATCH --error=train_run/stderr/airogs_improved_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=a40
#SBATCH --gres=gpu:1

echo "=============================================="
echo "AIROGS Training - Week 2 Improvements"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "Configuration:"
echo "  Memory: 32G"
echo "  CPUs: 8"
echo "  Time limit: 12:00:00"
echo "  GPU: 1x A40"
echo ""

# Load CUDA modules
echo "Loading CUDA modules..."
if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "No CUDA module, using system CUDA"
    module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module, using system cuDNN"
    echo "Loaded modules:"
    module list
else
    echo "Module command not available, using system CUDA/cuDNN"
fi
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

# Set GPU affinity for A40 (optimal for GPU0)
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Starting Week 2 Improved Training"
echo "=============================================="
echo "Improvements:"
echo "  - Focal Loss (γ=2.0, α=0.25)"
echo "  - CLAHE preprocessing"
echo "  - Enhanced augmentation"
echo "  - Datasets: 0, 1, 4"
echo ""

# Run training
python3 train_improved.py

PYTHON_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training completed with exit code: $PYTHON_EXIT_CODE"
echo "End time: $(date)"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training successful!"
    echo ""
    echo "📊 Results:"
    if [ -d "outputs_improved/models" ]; then
        echo "   Models:"
        ls -lh outputs_improved/models/*.keras 2>/dev/null | tail -5
    fi
    if [ -d "outputs_improved/logs" ]; then
        echo ""
        echo "   Logs:"
        ls -lh outputs_improved/logs/*.json 2>/dev/null | tail -5
    fi
else
    echo ""
    echo "❌ Training failed!"
fi

echo "=============================================="
EOFSLURM

# Submit job
echo ""
echo "🚀 Submitting SLURM job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]*')

if [ ! -z "$JOB_ID" ]; then
    echo ""
    echo "✅ Job submitted successfully!"
    echo ""
    echo "Job ID: $JOB_ID"
    echo "Output: train_run/output/airogs_improved_${JOB_ID}.out"
    echo "Errors: train_run/stderr/airogs_improved_${JOB_ID}.err"
    echo ""
    echo "📊 Useful commands:"
    echo "   squeue -u \$USER                                           # Check job status"
    echo "   tail -f train_run/output/airogs_improved_${JOB_ID}.out   # Follow training logs"
    echo "   tail -f train_run/stderr/airogs_improved_${JOB_ID}.err   # Follow error logs"
    echo "   scancel $JOB_ID                                            # Cancel job"
    echo ""
    echo "🕐 Estimated time: 4-6 hours"
    echo ""
    echo "After training completes, evaluate with:"
    echo "   ./submit_eval_improved.sh"
    echo ""
else
    echo ""
    echo "❌ Job submission failed"
    exit 1
fi

echo "============================================================"
