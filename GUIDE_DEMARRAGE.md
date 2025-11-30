# AIROGS Baseline - Guide de Démarrage Rapide

## 📁 Structure du Projet

Voici les fichiers que j'ai créés pour votre baseline AIROGS :

### Fichiers Python
1. **config.py** - Configuration centrale (paramètres, chemins, hyperparamètres)
2. **dataset.py** - Chargement et prétraitement des données
3. **model.py** - Architecture du modèle (EfficientNet-B0 baseline)
4. **evaluation.py** - Métriques d'évaluation (pAUC, sensibilité @ 95% spécificité)
5. **train.py** - Script d'entraînement principal
6. **inference.py** - Script d'inférence
7. **utils.py** - Fonctions utilitaires (visualisation, vérification des données)

### Scripts SLURM pour Cluster
1. **train_cluster.sh** - Script SBATCH pour entraînement complet (20 epochs, ~2h)
2. **train_cluster_quick.sh** - Script SBATCH pour test rapide (5 epochs, ~30min)

### Autres Fichiers
1. **requirements.txt** - Dépendances Python
2. **setup.sh** - Script de configuration automatique
3. **README.md** - Documentation complète

## 🚀 Utilisation

### Option 1 : Entraînement sur Cluster (RECOMMANDÉ)

```bash
# 1. Se connecter au cluster
ssh votre_username@cluster.address

# 2. Aller dans le répertoire du projet
cd /home/matteo/Bureau/FIB/cours/DLMA/MiccaiChallenge

# 3. Configurer l'environnement (première fois seulement)
./setup.sh

# 4. IMPORTANT : Modifier train_cluster.sh avec vos informations
#    - Nom de la partition GPU de votre cluster
#    - Modules à charger (python, cuda, cudnn)
#    - Votre email pour les notifications
nano train_cluster.sh

# 5. Soumettre le job
sbatch train_cluster.sh

# 6. Vérifier le statut
squeue -u $USER

# 7. Suivre les logs en temps réel
tail -f logs/airogs_baseline_*.out
```

### Option 2 : Entraînement Local

```bash
# 1. Installer les dépendances
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Lancer l'entraînement
python train.py
```

## ⚙️ Configuration du Cluster

**IMPORTANT** : Avant de soumettre votre job, vous devez adapter `train_cluster.sh` à votre cluster :

```bash
# Ouvrir le fichier
nano train_cluster.sh

# Modifier ces lignes selon votre cluster :
#SBATCH --partition=gpu          # ← Nom de votre partition GPU
#SBATCH --mail-user=votre@email  # ← Votre email

# Et ces lignes selon les modules disponibles :
module load python/3.9   # ← Version Python disponible
module load cuda/11.8    # ← Version CUDA disponible
module load cudnn/8.6    # ← Version cuDNN disponible
```

Pour connaître les modules disponibles sur votre cluster :
```bash
module avail python
module avail cuda
module avail cudnn
```

## 📊 Données Attendues

Le script s'attend à trouver :

```
data/
├── 0/                      # Images d'entraînement
│   ├── TRAIN000000.jpg
│   ├── TRAIN000001.jpg
│   └── ...
└── train_labels.csv        # Labels
```

Le fichier CSV doit contenir :
- `challenge_id` : ID de l'image (ex: TRAIN000000)
- `class` : Label (RG ou NRG)

## 🎯 Caractéristiques du Baseline

### Architecture
- **Backbone** : EfficientNet-B0 (pré-entraîné ImageNet)
- **Input** : Images 384×384 RGB
- **Output** : Classification binaire (RG vs NRG)
- **Tête de classification** : 3 couches denses avec dropout

### Gestion du Déséquilibre de Classes
- Ratio RG:NRG ≈ 1:30
- Solution : Loss pondérée (poids 30:1)

### Augmentation de Données
- Flip horizontal
- Rotation (±15°)
- Zoom (±10%)
- Ajustement de luminosité (±20%)

### Métriques d'Évaluation (Challenge AIROGS)
- **α (pAUC)** : AUC partielle à 90-100% spécificité
- **β** : Sensibilité à 95% de spécificité
- **γ (Kappa)** : Cohen's kappa pour gradabilité (placeholder)
- **δ (AUC)** : AUC pour détection d'images non-gradables (placeholder)

## 📈 Résultats Attendus

Pour ce baseline simple :
- **AUC** : 0.80-0.85
- **pAUC (90-100% spec)** : 0.75-0.80
- **Sensibilité @ 95% spec** : 0.65-0.75
- **Temps d'entraînement** : 1.5-2h (1 GPU)

*Note : Les gagnants du challenge ont atteint >0.90 pAUC avec des techniques avancées*

## 🔧 Personnalisation

### Modifier les Hyperparamètres

Éditez `config.py` :

```python
IMAGE_SIZE = 384          # Taille des images
BATCH_SIZE = 32           # Taille du batch
EPOCHS = 20               # Nombre d'epochs
LEARNING_RATE = 1e-4      # Taux d'apprentissage
MODEL_BACKBONE = "efficientnet-b0"  # Architecture
```

### Changer d'Architecture

Dans `train.py` ou via arguments :

```bash
python train.py --backbone resnet50
# ou
python train.py --backbone efficientnet-b3
```

## 📁 Sorties Générées

Après l'entraînement, vous trouverez :

```
outputs/
├── models/
│   ├── airogs_baseline_efficientnet-b0_YYYYMMDD_HHMMSS_best.h5
│   └── airogs_baseline_efficientnet-b0_YYYYMMDD_HHMMSS_final.h5
├── logs/
│   ├── airogs_baseline_efficientnet-b0_YYYYMMDD_HHMMSS_training.csv
│   └── airogs_baseline_efficientnet-b0_YYYYMMDD_HHMMSS_history.json
├── plots/
│   ├── *_roc.png
│   ├── *_confusion.png
│   └── *_distribution.png
└── airogs_baseline_efficientnet-b0_YYYYMMDD_HHMMSS_results.json
```

## 🔍 Inférence

```bash
# Image unique
python inference.py \
    --model outputs/models/votre_modele_best.h5 \
    --image chemin/vers/image.jpg

# Batch d'images
python inference.py \
    --model outputs/models/votre_modele_best.h5 \
    --image-dir chemin/vers/images/ \
    --output predictions.csv
```

## 🐛 Dépannage

### Out of Memory (OOM)
Réduisez le batch size dans `config.py` :
```python
BATCH_SIZE = 16  # ou 8
```

### Entraînement Trop Lent
Réduisez la taille des images :
```python
IMAGE_SIZE = 256
```

### Problème avec les Modules du Cluster
Listez les modules disponibles :
```bash
module avail
```

## 📚 Améliorations Possibles (Semaine 2+)

### Orientées Données
1. Augmentation avancée (MixUp, CutMix)
2. Détection et crop du disque optique
3. Pré-traitement spécifique fundus
4. Équilibrage avancé (Focal Loss, SMOTE)

### Orientées Modèle
1. Architectures plus grandes (EfficientNet-B3, ResNet-101)
2. Multi-task learning (glaucome + gradabilité)
3. Test-time augmentation
4. Estimation d'incertitude (MC Dropout)

### Orientées Entraînement
1. Optimiseurs avancés (AdamW, LAMB)
2. Learning rate schedules
3. Cross-validation K-fold

## 📞 Support

Pour toute question sur :
- Les métriques du challenge : voir `evaluation.py`
- L'architecture : voir `model.py`
- Le prétraitement : voir `dataset.py`
- La documentation complète : voir `README.md`

## ✅ Checklist Avant Soumission

- [ ] Données placées dans `data/0/` et `data/train_labels.csv`
- [ ] Script `train_cluster.sh` configuré avec vos paramètres cluster
- [ ] Environnement virtuel créé et dépendances installées
- [ ] Test rapide effectué avec `train_cluster_quick.sh`
- [ ] Logs vérifiés (pas d'erreurs)
- [ ] Résultats évalués (métriques affichées)

Bon courage ! 🚀

