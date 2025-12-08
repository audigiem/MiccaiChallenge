#!/bin/bash
#
# Script pour soumettre l'entraînement sur les datasets 0, 1, 4
#

echo "================================================"
echo "AIROGS Training - Datasets 0, 1, 4 (No Augmentation)"
echo "================================================"
echo ""

# Vérifier que les datasets existent
DATASETS=("dataset/0" "dataset/1" "dataset/4")
MISSING=0

echo "🔍 Vérification des datasets..."
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
    echo "❌ Erreur: Certains datasets sont manquants!"
    echo "   Vérifiez que vous êtes dans le bon répertoire."
    exit 1
fi

# Vérifier le fichier labels
echo ""
echo "🔍 Vérification du fichier labels..."
if [ ! -f "dataset/train_labels.csv" ]; then
    echo "   ❌ dataset/train_labels.csv - NOT FOUND"
    exit 1
else
    LINES=$(wc -l < "dataset/train_labels.csv")
    echo "   ✅ dataset/train_labels.csv - $LINES lignes"
fi

# Créer les répertoires nécessaires
echo ""
echo "📁 Création des répertoires..."
mkdir -p train_run/{stderr,output}
mkdir -p outputs/{models,logs,checkpoints}
echo "   ✅ Répertoires créés"

# Afficher la configuration
echo ""
echo "📋 Configuration de l'entraînement:"
echo "   Datasets: 0, 1, 4"
echo "   Augmentation: DISABLED"
echo "   Class weights: 1.0 (NRG) / 10.0 (RG)"
echo "   Batch size: 32"
echo "   Epochs: 20"
echo "   Image size: 384x384"
echo "   GPU: 1x RTX 6000"
echo "   Memory: 32GB"
echo "   Time limit: 8 hours"
echo ""

# Demander confirmation
read -p "Lancer l'entraînement? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Entraînement annulé"
    exit 1
fi

# Soumettre le job
echo ""
echo "🚀 Soumission du job SLURM..."
JOB_ID=$(sbatch train_datasets_014.sh | grep -o '[0-9]*')

if [ ! -z "$JOB_ID" ]; then
    echo ""
    echo "✅ Job soumis avec succès!"
    echo ""
    echo "Job ID: $JOB_ID"
    echo "Output: train_run/output/airogs_train_014_${JOB_ID}.out"
    echo "Errors: train_run/stderr/airogs_train_014_${JOB_ID}.err"
    echo ""
    echo "📊 Commandes utiles:"
    echo "   squeue -u \$USER                                    # Voir l'état du job"
    echo "   tail -f train_run/output/airogs_train_014_${JOB_ID}.out  # Suivre les logs"
    echo "   tail -f train_run/stderr/airogs_train_014_${JOB_ID}.err  # Suivre les erreurs"
    echo "   scancel $JOB_ID                                     # Annuler le job"
    echo ""
    echo "🕐 Temps estimé: 4-6 heures"
    echo ""
    echo "Une fois l'entraînement terminé, vérifiez:"
    echo "   ls -lh outputs/models/                              # Modèles générés"
    echo "   python3 inspect_model.py outputs/models/<model>.h5  # Inspecter le modèle"
    echo ""
else
    echo ""
    echo "❌ Échec de la soumission du job"
    exit 1
fi

echo "================================================"

