#!/bin/bash

# Evaluation job submission script
# Usage: ./submit_eval.sh [--model=path] [--data=path] [--time=02:00:00] [--mem=16G] [--cpus=4]

# Default values
MODEL_PATH="outputs/models/airogs_baseline_efficientnet-b0_20251202_132402_final.h5"
DATA_DIR="dataset/1"
LABELS_CSV="dataset/train_labels.csv"
BATCH_SIZE=32
OUTPUT_DIR="evaluation_results"
MEMORY="16G"
CPUS="4"
TIME_LIMIT="02:00:00"
ENV_PATH="~/MiccaiChallenge/bin/activate"
JOB_NAME="airogs_eval"

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
        --output=*)
            OUTPUT_DIR="${1#*=}"
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
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        --batch=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model=path] [--data=path] [--labels=path] [--output=path] [--time=02:00:00] [--mem=16G] [--cpus=4] [--batch=32]"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p eval_run/{stderr,output}

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="eval_run/slurm_eval_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=eval_run/output/${JOB_NAME}_${TIMESTAMP}.out
#SBATCH --error=eval_run/stderr/${JOB_NAME}_${TIMESTAMP}.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1

# Log job start
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] AIROGS Challenge - Evaluation on Cluster"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Job ID: \\$SLURM_JOB_ID"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Node: \\$SLURM_NODELIST"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Configuration: MEM=${MEMORY}, CPUS=${CPUS}, TIME=${TIME_LIMIT}"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] GPU: \\$CUDA_VISIBLE_DEVICES"
echo ""

# Load CUDA modules
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Loading CUDA modules..."
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "No CUDA module found, using system CUDA"
module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module found, using system cuDNN"

# Display loaded modules
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Loaded modules:"
module list
echo ""

# Vérifier la disponibilité du GPU
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU availability..."
nvidia-smi
echo ""

# Activate environment
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Activating virtual environment..."
source ${ENV_PATH}
export OMP_NUM_THREADS=${CPUS}
export TF_CPP_MIN_LOG_LEVEL=1

echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation..."
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Model: ${MODEL_PATH}"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Data dir: ${DATA_DIR}"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Labels: ${LABELS_CSV}"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Output: ${OUTPUT_DIR}"
echo ""

# Exécuter l'évaluation
python3 evaluation.py \\
    --model-path="${MODEL_PATH}" \\
    --data-dir="${DATA_DIR}" \\
    --labels-csv="${LABELS_CSV}" \\
    --batch-size=${BATCH_SIZE} \\
    --output-dir="${OUTPUT_DIR}"

PYTHON_EXIT_CODE=\\$?

echo ""
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed with exit code: \\$PYTHON_EXIT_CODE"
echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] End time: \\$(date)"

# Final summary
if [ \\$PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed successfully!"

    # List generated files
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Generated files:"
        ls -lh "${OUTPUT_DIR}"
    fi
else
    echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] Evaluation failed!"
fi

echo "[\\$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
EOF

# Submit the job
echo "Submitting evaluation job to SLURM..."
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_DIR"
echo "  Labels: $LABELS_CSV"
echo "  Output: $OUTPUT_DIR"
echo "  Resources: MEM=$MEMORY, CPUS=$CPUS, TIME=$TIME_LIMIT"
echo ""

JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]*')

if [ ! -z "$JOB_ID" ]; then
    echo "✅ Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Output: eval_run/output/${JOB_NAME}_${TIMESTAMP}.out"
    echo "Errors: eval_run/stderr/${JOB_NAME}_${TIMESTAMP}.err"
    echo ""
    echo "Monitor your job with:"
    echo "   squeue -u \$USER                                       # Check job status"
    echo "   tail -f eval_run/output/${JOB_NAME}_${TIMESTAMP}.out  # Follow output"
    echo "   tail -f eval_run/stderr/${JOB_NAME}_${TIMESTAMP}.err  # Follow errors"
    echo "   scancel $JOB_ID                                        # Cancel if needed"
    echo ""
    echo "You can now safely close this terminal - the job will continue running!"
else
    echo "❌ Failed to submit job"
    exit 1
fi

