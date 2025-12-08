#!/bin/bash
#
# Script pour diviser train_labels.csv en plusieurs fichiers
# À exécuter sur le cluster
#

echo "================================================"
echo "AIROGS Challenge - Séparation des labels"
echo "================================================"

# Configuration par défaut
LABELS_CSV="${1:-dataset/train_labels.csv}"
DATASET_DIR="${2:-dataset/datasetPart1}"
OUTPUT_DIR="${3:-dataset}"

echo "Labels CSV: ${LABELS_CSV}"
echo "Dataset directory: ${DATASET_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Vérifier que les fichiers existent
if [ ! -f "${LABELS_CSV}" ]; then
    echo "❌ Erreur: Fichier labels introuvable: ${LABELS_CSV}"
    exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "❌ Erreur: Répertoire dataset introuvable: ${DATASET_DIR}"
    exit 1
fi

# Activer l'environnement virtuel si nécessaire
if [ -n "${VIRTUAL_ENV}" ]; then
    echo "✅ Environnement virtuel déjà activé: ${VIRTUAL_ENV}"
elif [ -f "venv/bin/activate" ]; then
    echo "🔄 Activation de l'environnement virtuel..."
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    echo "🔄 Activation de l'environnement virtuel..."
    source ../venv/bin/activate
fi

# Exécuter le script Python
echo ""
echo "🚀 Démarrage de la séparation des labels..."
echo ""

python3 split_labels.py \
    --labels "${LABELS_CSV}" \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Séparation terminée avec succès!"
else
    echo "❌ Erreur lors de la séparation (code: ${EXIT_CODE})"
fi

echo "================================================"

exit ${EXIT_CODE}

