#!/bin/bash
#
# Training script for Week 2 improvements V2 - OPTIMIZED
# Based on working v1 structure with optimized parameters
#

echo "============================================================"
echo "AIROGS Week 2 Improved Training V2 - OPTIMIZED"
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
echo "📋 Week 2 Improved V2 Configuration (OPTIMIZED):"
echo "   Datasets: 0, 1, 4"
echo "   🆕 OPTIMIZED Focal Loss: gamma=1.5, alpha=0.75"
echo "   🆕 CLAHE Preprocessing: ENABLED"
echo "   🆕 Moderate Augmentation: ENABLED"
echo "   🚀 Epochs: 15 (REDUCED from 30)"
echo "   🚀 Learning Rate: 1e-4 (INCREASED)"
echo "   Batch size: 32"
echo "   Image size: 384x384"
echo "   GPU: 1x NVIDIA RTX 6000 (24GB VRAM)"
echo "   Partition: rtx6000"
echo "   Memory: 32GB"
echo "   CPUs: 8"
echo "   Time limit: 8 hours"
echo ""

# Confirm
read -p "Launch optimized v2 training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 1
fi

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="train_run/slurm_train_improved_v2_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << 'EOFSLURM'
#!/bin/bash
#SBATCH --job-name=airogs_improved_v2
#SBATCH --output=train_run/output/airogs_improved_v2_%j.out
#SBATCH --error=train_run/stderr/airogs_improved_v2_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1

echo "=============================================="
echo "AIROGS Training - Week 2 Improvements V2"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "Configuration:"
echo "  Memory: 32G"
echo "  CPUs: 8"
echo "  Time limit: 08:00:00"
echo "  GPU: 1x RTX 6000"
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

# Set GPU affinity for RTX 6000
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Starting Week 2 Improved V2 Training (OPTIMIZED)"
echo "=============================================="
echo "Optimizations:"
echo "  - REDUCED epochs: 15 (was 30)"
echo "  - INCREASED learning rate: 1e-4 (was 5e-5)"
echo "  - OPTIMIZED focal loss: gamma=1.5, alpha=0.75"
echo "  - Moderate augmentation (not too aggressive)"
echo "  - CLAHE preprocessing"
echo "  - Datasets: 0, 1, 4"
echo ""

# Run training
python3 train_improved_v2.py

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
    echo "Output: train_run/output/airogs_improved_v2_${JOB_ID}.out"
    echo "Errors: train_run/stderr/airogs_improved_v2_${JOB_ID}.err"
    echo ""
    echo "📊 Useful commands:"
    echo "   squeue -u \$USER                                              # Check job status"
    echo "   tail -f train_run/output/airogs_improved_v2_${JOB_ID}.out   # Follow training logs"
    echo "   tail -f train_run/stderr/airogs_improved_v2_${JOB_ID}.err   # Follow error logs"
    echo "   scancel $JOB_ID                                               # Cancel job"
    echo ""
    echo "🕐 Estimated time: 3-4 hours (OPTIMIZED - was 6+ hours)"
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

# Check exit status
EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Check training logs in outputs/logs/"
    echo "  2. Best model saved as: outputs/models/*_best.keras"
    echo "  3. Run evaluation: python evaluation_improved.py --model-path outputs/models/*_best.keras"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check stderr file for details"
fi
echo "=============================================="
echo "End time: $(date)"
echo ""

exit $EXIT_CODE
