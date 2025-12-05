#!/bin/bash
#SBATCH --job-name=airogs_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Configuration
MODEL_PATH="firstModel.h5"
DATA_DIR="5"
LABELS_CSV="train_labels.csv"
BATCH_SIZE=32
OUTPUT_DIR="evaluation_results"

# Activation de l'environnement virtuel (si nécessaire)
# source /path/to/venv/bin/activate

# Afficher les informations système
echo "=========================================="
echo "AIROGS Challenge - Evaluation on Cluster"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Vérifier la disponibilité du GPU
nvidia-smi

echo ""
echo "=========================================="
echo "Starting evaluation..."
echo "=========================================="
echo ""

# Exécuter l'évaluation
python evaluation.py \
    --model-path="${MODEL_PATH}" \
    --data-dir="${DATA_DIR}" \
    --labels-csv="${LABELS_CSV}" \
    --batch-size=${BATCH_SIZE} \
    --output-dir="${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

