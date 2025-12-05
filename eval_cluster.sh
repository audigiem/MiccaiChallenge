#!/bin/bash
#SBATCH --job-name=airogs_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000

# Configuration
MODEL_PATH="outputs/models/airogs_baseline_efficientnet-b0_20251202_132402_final.h5"
DATA_DIR="dataset/1"
LABELS_CSV="dataset/train_labels.csv"
BATCH_SIZE=32
OUTPUT_DIR="evaluation_results"

# Log job start
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] AIROGS Challenge - Evaluation on Cluster"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ID: $SLURM_JOB_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Node: $SLURM_NODELIST"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration: MEM=16G, CPUS=4, TIME=02:00:00"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load CUDA modules (adjust versions if needed)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading CUDA modules..."
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "No CUDA module found, using system CUDA"
module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module found, using system cuDNN"

# Display loaded modules
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded modules:"
module list
echo ""

# Vérifier la disponibilité du GPU
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU availability..."
nvidia-smi
echo ""

# Activation de l'environnement virtuel
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating virtual environment..."
source ~/MiccaiChallenge/bin/activate
export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model: ${MODEL_PATH}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Data dir: ${DATA_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Labels: ${LABELS_CSV}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output: ${OUTPUT_DIR}"
echo ""

# Exécuter l'évaluation
python3 evaluation.py \
    --model-path="${MODEL_PATH}" \
    --data-dir="${DATA_DIR}" \
    --labels-csv="${LABELS_CSV}" \
    --batch-size=${BATCH_SIZE} \
    --output-dir="${OUTPUT_DIR}"

PYTHON_EXIT_CODE=$?

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed with exit code: $PYTHON_EXIT_CODE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] End time: $(date)"

# Final summary
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed successfully!"

    # List generated files
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated files:"
        ls -lh "${OUTPUT_DIR}"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation failed!"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="

