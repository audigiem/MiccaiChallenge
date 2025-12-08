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

# Auto-detect labels file based on dataset directory
# If DATA_DIR is like "dataset/1" or "dataset/datasetPart1/1", extract the dataset number
if [[ "$LABELS_CSV" == "dataset/train_labels.csv" ]]; then
    # Extract dataset identifier (e.g., "1" from "dataset/1" or "dataset/datasetPart1/1")
    DATASET_ID=$(basename "$DATA_DIR")
    DATASET_PARENT=$(dirname "$DATA_DIR")

    # Check if a specific labels file exists for this dataset
    if [[ -f "dataset/train_labels_${DATASET_ID}.csv" ]]; then
        LABELS_CSV="dataset/train_labels_${DATASET_ID}.csv"
        echo "ℹ️  Auto-detected labels file: $LABELS_CSV"
    elif [[ -f "${DATASET_PARENT}/train_labels_${DATASET_ID}.csv" ]]; then
        LABELS_CSV="${DATASET_PARENT}/train_labels_${DATASET_ID}.csv"
        echo "ℹ️  Auto-detected labels file: $LABELS_CSV"
    else
        echo "⚠️  Warning: No specific labels file found for dataset '$DATASET_ID'"
        echo "    Using default: $LABELS_CSV"
        echo "    You may need to run: ./run_split_labels.sh first"
    fi
fi

# Create directories
mkdir -p eval_run/{stderr,output}

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="eval_run/slurm_eval_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --output=eval_run/output/JOBNAME_PLACEHOLDER_TIMESTAMP_PLACEHOLDER.out
#SBATCH --error=eval_run/stderr/JOBNAME_PLACEHOLDER_TIMESTAMP_PLACEHOLDER.err
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEMORY_PLACEHOLDER
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1

# Log job start
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] AIROGS Challenge - Evaluation on Cluster"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ID: $SLURM_JOB_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Node: $SLURM_NODELIST"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration: MEM=MEMORY_PLACEHOLDER, CPUS=CPUS_PLACEHOLDER, TIME=TIME_PLACEHOLDER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load CUDA modules
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading CUDA modules..."
if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "No CUDA module found, using system CUDA"
    module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module found, using system cuDNN"
else
    echo "Module command not available, using system CUDA/cuDNN"
fi

# Display loaded modules
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded modules:"
if command -v module &> /dev/null; then
    module list
else
    echo "Module system not available"
fi
echo ""

# Vérifier la disponibilité du GPU
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU availability..."
nvidia-smi
echo ""

# Activate environment
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating virtual environment..."
source ENV_PATH_PLACEHOLDER
export OMP_NUM_THREADS=CPUS_PLACEHOLDER
export TF_CPP_MIN_LOG_LEVEL=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model: MODEL_PATH_PLACEHOLDER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Data dir: DATA_DIR_PLACEHOLDER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Labels: LABELS_CSV_PLACEHOLDER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output: OUTPUT_DIR_PLACEHOLDER"
echo ""

# Exécuter l'évaluation
python3 evaluation.py \
    --model-path="MODEL_PATH_PLACEHOLDER" \
    --data-dir="DATA_DIR_PLACEHOLDER" \
    --labels-csv="LABELS_CSV_PLACEHOLDER" \
    --batch-size=BATCH_SIZE_PLACEHOLDER \
    --output-dir="OUTPUT_DIR_PLACEHOLDER"

PYTHON_EXIT_CODE=$?

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed with exit code: $PYTHON_EXIT_CODE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] End time: $(date)"

# Final summary
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed successfully!"

    # List generated files
    if [ -d "OUTPUT_DIR_PLACEHOLDER" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated files:"
        ls -lh "OUTPUT_DIR_PLACEHOLDER"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation failed!"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
EOF

# Replace placeholders with actual values
sed -i "s|JOBNAME_PLACEHOLDER|${JOB_NAME}|g" "$SLURM_SCRIPT"
sed -i "s|TIMESTAMP_PLACEHOLDER|${TIMESTAMP}|g" "$SLURM_SCRIPT"
sed -i "s|TIME_PLACEHOLDER|${TIME_LIMIT}|g" "$SLURM_SCRIPT"
sed -i "s|CPUS_PLACEHOLDER|${CPUS}|g" "$SLURM_SCRIPT"
sed -i "s|MEMORY_PLACEHOLDER|${MEMORY}|g" "$SLURM_SCRIPT"
sed -i "s|ENV_PATH_PLACEHOLDER|${ENV_PATH}|g" "$SLURM_SCRIPT"
sed -i "s|MODEL_PATH_PLACEHOLDER|${MODEL_PATH}|g" "$SLURM_SCRIPT"
sed -i "s|DATA_DIR_PLACEHOLDER|${DATA_DIR}|g" "$SLURM_SCRIPT"
sed -i "s|LABELS_CSV_PLACEHOLDER|${LABELS_CSV}|g" "$SLURM_SCRIPT"
sed -i "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" "$SLURM_SCRIPT"
sed -i "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" "$SLURM_SCRIPT"

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

