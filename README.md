# AIROGS Glaucoma Detection - Baseline Implementation

This is a baseline implementation for the AIROGS challenge on automatic detection of referable glaucoma from fundus images.

## 📋 Project Structure

```
.
├── config.py                 # Configuration parameters
├── dataset.py                # Dataset loading and preprocessing
├── model.py                  # Model architecture and compilation
├── evaluation.py             # Evaluation metrics and visualization
├── train.py                  # Main training script
├── inference.py              # Inference script
├── train_cluster.sh          # SLURM script for cluster training (full)
├── train_cluster_quick.sh    # SLURM script for quick testing
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   ├── 0/                    # Training images
│   └── train_labels.csv      # Training labels
└── outputs/                  # Output directory
    ├── models/               # Saved models
    ├── logs/                 # Training logs
    └── plots/                # Evaluation plots
```

## 🎯 Baseline Approach

### Model Architecture
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: 384×384 RGB fundus images
- **Output**: Binary classification (RG vs NRG)
- **Classification head**: 3 dense layers with dropout

### Key Features
1. **Simple preprocessing**: Resize to 384×384, normalize to [0,1]
2. **Data augmentation**: Horizontal flip, rotation, zoom, brightness
3. **Class imbalance handling**: Weighted loss (1:30 ratio)
4. **Fast training**: ~2 hours on single GPU (Colab compatible)

### Evaluation Metrics (AIROGS Challenge)
- **α (pAUC)**: Partial AUC at 90-100% specificity
- **β**: Sensitivity at 95% specificity
- **γ (Kappa)**: Cohen's kappa for gradability (placeholder in baseline)
- **δ (AUC)**: AUC for ungradability detection (placeholder in baseline)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your data as follows:
```
data/
├── 0/                    # Folder containing training images
│   ├── TRAIN000000.jpg
│   ├── TRAIN000001.jpg
│   └── ...
└── train_labels.csv      # CSV with columns: challenge_id, class
```

### 3. Train Locally

```bash
# Basic training
python train.py

# Custom parameters
python train.py \
    --data-dir ./data \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --image-size 384 \
    --backbone efficientnet-b0
```

### 4. Train on Cluster with SLURM

```bash
# Make scripts executable
chmod +x train_cluster.sh train_cluster_quick.sh

# Quick test (5 epochs, ~30 minutes)
sbatch train_cluster_quick.sh

# Full training (20 epochs, ~2 hours)
sbatch train_cluster.sh

# Check job status
squeue -u $USER

# View output
tail -f logs/airogs_baseline_*.out
```

### 5. Run Inference

```bash
# Single image
python inference.py \
    --model outputs/models/airogs_baseline_efficientnet-b0_best.h5 \
    --image path/to/image.jpg

# Batch inference
python inference.py \
    --model outputs/models/airogs_baseline_efficientnet-b0_best.h5 \
    --image-dir path/to/images/ \
    --output predictions.csv
```

## 📊 Expected Results

Based on the baseline approach, you can expect:

- **AUC**: 0.80-0.85
- **Partial AUC (90-100% spec)**: 0.75-0.80
- **Sensitivity @ 95% specificity**: 0.65-0.75
- **Training time**: 1.5-2 hours (single GPU)

*Note: These are baseline results. The challenge winners achieved >0.90 pAUC through advanced techniques.*

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_BACKBONE = "efficientnet-b0"  # or "resnet50", "efficientnet-b3"

# Class imbalance (RG:NRG ≈ 1:30)
CLASS_WEIGHTS = {0: 1.0, 1: 30.0}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
```

## 📈 Improvements (Week 2+)

Potential enhancements to explore:

### Data-focused improvements:
1. **Advanced augmentation**: MixUp, CutMix, RandAugment
2. **Better class balancing**: Focal loss, SMOTE
3. **Image preprocessing**: Optic disc detection and cropping
4. **Multi-resolution training**: Different image sizes

### Model improvements:
1. **Larger backbone**: EfficientNet-B3, ResNet-101
2. **Gradability head**: Multi-task learning for robustness
3. **Test-time augmentation**: Average predictions over augmented versions
4. **Uncertainty estimation**: Monte-Carlo dropout

### Training improvements:
1. **Advanced optimizers**: AdamW, LAMB
2. **Learning rate schedules**: Cosine annealing, warmup
3. **Cross-validation**: K-fold for better generalization

## 📝 SLURM Script Customization

Edit `train_cluster.sh` for your cluster:

```bash
#SBATCH --partition=gpu          # Your GPU partition name
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH --time=12:00:00          # Max time
#SBATCH --mem=32G                # Memory
#SBATCH --mail-user=your@email   # Your email

# Load your cluster's modules
module load python/3.9
module load cuda/11.8
module load cudnn/8.6
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # or 8

# Reduce image size
IMAGE_SIZE = 256
```

### Slow Training
```python
# Enable mixed precision (should be enabled by default)
USE_MIXED_PRECISION = True

# Use fewer augmentations
AUGMENTATION = {"horizontal_flip": True}
```

### Class Imbalance Issues
```python
# Increase weight for minority class
CLASS_WEIGHTS = {0: 1.0, 1: 50.0}

# Or use focal loss (implement in model.py)
```

## 📚 References

1. AIROGS Challenge: https://airogs.grand-challenge.org/
2. Challenge Paper: https://doi.org/10.1167/iovs.63.8.3
3. EfficientNet: https://arxiv.org/abs/1905.11946
4. Medical Image Analysis: Best practices for fundus image analysis

## 📄 License

This is an educational project for the DLMA course.

## 👥 Author

Matteo - FIB/UPC DLMA Course

## 🙏 Acknowledgments

- AIROGS challenge organizers
- TensorFlow/Keras community
- Course instructors

