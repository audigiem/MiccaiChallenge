#!/bin/bash
#
# Script pour évaluer le modèle sur tous les datasets
# Usage: ./eval_all_datasets.sh [model_path]
#

MODEL_PATH="${1:-outputs/models/airogs_baseline_efficientnet-b0_20251202_132402_final.h5}"
DATASET_BASE="dataset/datasetPart1"
BATCH_SIZE=32
MEMORY="16G"
CPUS="4"
TIME_LIMIT="02:00:00"

echo "================================================"
echo "AIROGS Challenge - Évaluation Multi-Datasets"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "Dataset base: $DATASET_BASE"
echo ""

# Vérifier que le modèle existe
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Erreur: Modèle introuvable: $MODEL_PATH"
    exit 1
fi

# Vérifier que le répertoire de base existe
if [ ! -d "$DATASET_BASE" ]; then
    echo "❌ Erreur: Répertoire dataset introuvable: $DATASET_BASE"
    exit 1
fi

# Trouver tous les sous-dossiers
DATASETS=($(find "$DATASET_BASE" -mindepth 1 -maxdepth 1 -type d | sort))

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "❌ Erreur: Aucun sous-dossier trouvé dans $DATASET_BASE"
    exit 1
fi

echo "📁 ${#DATASETS[@]} datasets trouvés:"
for dataset in "${DATASETS[@]}"; do
    echo "   - $(basename "$dataset")"
done
echo ""

# Confirmer
read -p "Lancer l'évaluation sur tous ces datasets? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Annulé"
    exit 1
fi

echo ""
echo "🚀 Lancement des jobs d'évaluation..."
echo ""

# Soumettre un job pour chaque dataset
JOB_IDS=()
for dataset_path in "${DATASETS[@]}"; do
    dataset_name=$(basename "$dataset_path")
    labels_file="dataset/train_labels_${dataset_name}.csv"
    output_dir="evaluation_results_${dataset_name}"

    echo "📊 Dataset: $dataset_name"
    echo "   Data: $dataset_path"
    echo "   Labels: $labels_file"
    echo "   Output: $output_dir"

    # Vérifier que le fichier de labels existe
    if [ ! -f "$labels_file" ]; then
        echo "   ⚠️  Warning: Labels file not found: $labels_file"
        echo "   ⏭️  Skipping this dataset"
        echo ""
        continue
    fi

    # Soumettre le job
    ./submit_eval.sh \
        --model="$MODEL_PATH" \
        --data="$dataset_path" \
        --labels="$labels_file" \
        --output="$output_dir" \
        --batch="$BATCH_SIZE" \
        --mem="$MEMORY" \
        --cpus="$CPUS" \
        --time="$TIME_LIMIT" > /tmp/eval_submit_$$.tmp

    # Extraire le Job ID
    JOB_ID=$(grep "Job ID:" /tmp/eval_submit_$$.tmp | cut -d: -f2 | xargs)
    if [ ! -z "$JOB_ID" ]; then
        JOB_IDS+=("$JOB_ID")
        echo "   ✅ Job submitted: $JOB_ID"
    else
        echo "   ❌ Failed to submit job"
    fi
    echo ""
done

rm -f /tmp/eval_submit_$$.tmp

echo "================================================"
echo "✅ ${#JOB_IDS[@]} jobs soumis"
echo ""

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs: ${JOB_IDS[@]}"
    echo ""
    echo "Commandes utiles:"
    echo "   squeue -u \$USER                      # Voir tous vos jobs"
    echo "   squeue -j ${JOB_IDS[0]}              # Voir un job spécifique"
    echo "   scancel ${JOB_IDS[@]}                # Annuler tous les jobs"
    echo "   ls -la evaluation_results_*/         # Voir les résultats"
fi

echo "================================================"

