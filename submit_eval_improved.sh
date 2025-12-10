#!/bin/bash
#
# Evaluation script for Week 2 improvements with TTA
# Optimized for NVIDIA A40 GPUs (ampere partition)
# Usage: ./submit_eval_improved.sh [--model=path] [--tta] [--clahe]
#

# Default values
MODEL_PATH=""
DATA_DIR="dataset/5"
LABELS_CSV="dataset/train_labels_5.csv"
USE_TTA="--tta"
USE_CLAHE=""
MEMORY="24G"
CPUS="6"
TIME_LIMIT="03:00:00"
ENV_PATH="~/MiccaiChallenge/bin/activate"
JOB_NAME="airogs_eval_improved"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --data=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        --labels=*)
            LABELS_CSV="${1#*=}"
            shift
            ;;
        --tta)
            USE_TTA="--tta"
            shift
            ;;
        --no-tta)
            USE_TTA=""
            shift
            ;;
        --clahe)
            USE_CLAHE="--clahe"
            shift
            ;;
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        --mem=*)
            MEMORY="${1#*=}"
            shift
            ;;
        --cpus=*)
            CPUS="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model=path] [--data=path] [--labels=path] [--tta|--no-tta] [--clahe] [--time=HH:MM:SS] [--mem=XG] [--cpus=N]"
            exit 1
            ;;
    esac
done

# Auto-detect latest improved model if not specified
if [ -z "$MODEL_PATH" ]; then
    echo "🔍 Auto-detecting latest improved model..."
    
    # Try to find the best model first
    BEST_MODEL=$(ls -t outputs_improved/models/*_best.keras 2>/dev/null | head -1)
    
    if [ ! -z "$BEST_MODEL" ]; then
        MODEL_PATH="$BEST_MODEL"
        echo "   Found best model: $MODEL_PATH"
    else
        # Fallback to final model
        FINAL_MODEL=$(ls -t outputs_improved/models/*_final.keras 2>/dev/null | head -1)
        if [ ! -z "$FINAL_MODEL" ]; then
            MODEL_PATH="$FINAL_MODEL"
            echo "   Found final model: $MODEL_PATH"
        else
            echo "   ❌ No improved model found in outputs_improved/models/"
            echo ""
            echo "Please train a model first:"
            echo "   ./submit_train_improved.sh"
            echo ""
            echo "Or specify a model manually:"
            echo "   ./submit_eval_improved.sh --model=path/to/model.keras"
            exit 1
        fi
    fi
fi

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found: $MODEL_PATH"
    exit 1
fi

# Create directories
mkdir -p eval_run/{stderr,output}

echo "============================================================"
echo "AIROGS Week 2 Improved Evaluation"
echo "============================================================"
echo ""
echo "📋 Configuration:"
echo "   Model: $MODEL_PATH"
echo "   Data directory: $DATA_DIR"
echo "   Labels CSV: $LABELS_CSV"
echo "   Test-Time Augmentation: ${USE_TTA:+ENABLED}"
echo "   CLAHE Preprocessing: ${USE_CLAHE:+ENABLED}"
echo "   GPU: 1x NVIDIA RTX 6000 (24GB VRAM)"
echo "   Partition: rtx6000"
echo "   Memory: $MEMORY"
echo "   CPUs: $CPUS"
echo "   Time limit: $TIME_LIMIT"
echo ""

# Confirm
read -p "Launch improved evaluation? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Evaluation cancelled"
    exit 1
fi

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="eval_run/slurm_eval_improved_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << EOFSLURM
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=eval_run/output/${JOB_NAME}_${TIMESTAMP}_%j.out
#SBATCH --error=eval_run/stderr/${JOB_NAME}_${TIMESTAMP}_%j.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1

echo "=============================================="
echo "AIROGS Evaluation - Week 2 Improvements"
echo "=============================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo ""

echo "Configuration:"
echo "  Memory: ${MEMORY}"
echo "  CPUs: ${CPUS}"
echo "  Time limit: ${TIME_LIMIT}"
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
source ${ENV_PATH}
export OMP_NUM_THREADS=${CPUS}
export TF_CPP_MIN_LOG_LEVEL=0
export PYTHONUNBUFFERED=1

# Set GPU affinity for RTX 6000
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Starting Improved Evaluation"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Data directory: ${DATA_DIR}"
echo "Labels CSV: ${LABELS_CSV}"
echo "TTA: ${USE_TTA:+ENABLED}"
echo "CLAHE: ${USE_CLAHE:+ENABLED}"
echo ""

# Run evaluation with unbuffered output
python3 -u evaluation_improved.py \
    --model-path="${MODEL_PATH}" \
    --data-dir="${DATA_DIR}" \
    --labels-csv="${LABELS_CSV}" \
    ${USE_TTA} \
    ${USE_CLAHE}

PYTHON_EXIT_CODE=\$?

echo ""
echo "=============================================="
echo "Evaluation completed with exit code: \$PYTHON_EXIT_CODE"
echo "End time: \$(date)"

if [ \$PYTHON_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Evaluation successful!"
    echo ""
    echo "📊 Results saved in model directory"
    MODEL_DIR=\$(dirname "${MODEL_PATH}")
    if [ -d "\$MODEL_DIR" ]; then
        echo "   Results files:"
        ls -lht "\$MODEL_DIR"/improved_evaluation_*.json 2>/dev/null | head -3
    fi
    echo ""
    echo "📈 Comparison with baseline:"
    echo "   Baseline pAUC: 0.6246"
    echo "   Baseline Sensitivity @ 95% spec: 0.6778"
    echo ""
    echo "   Check the output above for improved metrics!"
else
    echo ""
    echo "❌ Evaluation failed!"
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
    echo "Output: eval_run/output/${JOB_NAME}_${TIMESTAMP}_${JOB_ID}.out"
    echo "Errors: eval_run/stderr/${JOB_NAME}_${TIMESTAMP}_${JOB_ID}.err"
    echo ""
    echo "📊 Useful commands:"
    echo "   squeue -u \$USER                                                      # Check job status"
    echo "   tail -f eval_run/output/${JOB_NAME}_${TIMESTAMP}_${JOB_ID}.out      # Follow evaluation logs"
    echo "   scancel $JOB_ID                                                       # Cancel job"
    echo ""
    echo "🕐 Estimated time: 30-60 minutes"
    echo ""
else
    echo ""
    echo "❌ Job submission failed"
    exit 1
fi

echo "============================================================"
