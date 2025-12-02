#!/bin/bash
#SBATCH --job-name=tensorflow_gpu_test
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --gres=shard:1          # Demande 1 partition de GPU
#SBATCH --cpus-per-task=4       # 4 CPUs comme recommandé
#SBATCH --mem=8GB               # 8 GB de RAM
#SBATCH --time=00:10:00         # 10 minutes max

echo "Début du job Slurm: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocation GPU: $SLURM_GPUS"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Charger l'environnement Python/TensorFlow si nécessaire
# module load python/3.10 cuda/11.8  # À adapter selon votre cluster

# Activer l'environnement virtuel (si vous en avez un)
 source ~/.MiccaiChallenge/bin/activate

# Exécuter le script de test
python test_gpu.py

echo "Fin du job Slurm: $(date)"