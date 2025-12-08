#!/bin/bash
#
# Script pour vérifier l'état d'un modèle entraîné
#

MODEL_PATH="${1}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./check_model.sh <model_path>"
    echo ""
    echo "Exemple:"
    echo "  ./check_model.sh outputs/models/airogs_baseline_efficientnet-b0_20251202_132402_final.h5"
    exit 1
fi

echo "=============================================="
echo "Vérification du Modèle"
echo "=============================================="
echo ""

# Vérifier si le fichier existe
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Fichier modèle introuvable: $MODEL_PATH"
    exit 1
fi

# Informations sur le fichier
echo "📁 Fichier Modèle:"
echo "   Chemin: $MODEL_PATH"
ls -lh "$MODEL_PATH" | awk '{print "   Taille:", $5}'
echo "   Date: $(stat -c %y "$MODEL_PATH" 2>/dev/null || stat -f %Sm "$MODEL_PATH" 2>/dev/null || echo "N/A")"
echo ""

# Vérifier le répertoire de sortie
MODEL_DIR=$(dirname "$MODEL_PATH")
OUTPUT_DIR=$(dirname "$MODEL_DIR")

echo "📊 Fichiers dans le répertoire de sortie:"
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    ls -lh "$OUTPUT_DIR"/*.h5 2>/dev/null | while read line; do
        echo "   $line"
    done
    echo ""
fi

# Vérifier s'il y a des checkpoints
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "💾 Checkpoints trouvés:"
    ls -lh "$CHECKPOINT_DIR"/*.h5 2>/dev/null | head -5 | while read line; do
        echo "   $line"
    done
    CHECKPOINT_COUNT=$(ls "$CHECKPOINT_DIR"/*.h5 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 5 ]; then
        echo "   ... et $(($CHECKPOINT_COUNT - 5)) autres"
    fi
    echo ""
fi

# Vérifier les logs d'entraînement
LOGS_DIR="$OUTPUT_DIR/logs"
if [ -d "$LOGS_DIR" ]; then
    echo "📝 Logs d'entraînement:"
    LOG_FILES=$(ls -t "$LOGS_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILES" ]; then
        LOG_FILE="$LOG_FILES"
        echo "   Dernier fichier log: $(basename "$LOG_FILE")"
        echo ""
        echo "   Dernières lignes:"
        tail -20 "$LOG_FILE" | sed 's/^/      /'
    else
        echo "   ⚠️  Aucun fichier log trouvé"
    fi
    echo ""
fi

# Vérifier les fichiers d'historique
HISTORY_FILE="$OUTPUT_DIR/training_history.csv"
if [ -f "$HISTORY_FILE" ]; then
    echo "📈 Historique d'entraînement trouvé:"
    echo "   Fichier: $HISTORY_FILE"
    EPOCHS=$(wc -l < "$HISTORY_FILE")
    echo "   Époques: $((EPOCHS - 1))"  # -1 pour l'en-tête

    # Afficher les dernières époques
    echo ""
    echo "   Dernières époques:"
    tail -5 "$HISTORY_FILE" | column -t -s',' | sed 's/^/      /'
    echo ""
else
    echo "⚠️  Fichier d'historique introuvable: $HISTORY_FILE"
    echo ""
fi

# Inspection du modèle avec Python
echo "🔍 Inspection détaillée du modèle:"
echo ""

if command -v python3 &> /dev/null; then
    python3 inspect_model.py "$MODEL_PATH"
else
    echo "   ⚠️  Python3 non disponible, inspection impossible"
fi

echo ""
echo "=============================================="
echo "✅ Vérification terminée"
echo "=============================================="

