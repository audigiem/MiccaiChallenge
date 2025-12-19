"""
Configuration for V5 - FINAL OPTIMIZED VERSION
Based on V4 training analysis - addressing validation overfitting
"""

import os

# Get absolute paths
V5_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(V5_DIR)

# Paths relative to parent directory
DATA_DIR = os.path.join(PARENT_DIR, "dataset")
TRAIN_IMAGES_DIR = [
    os.path.join(DATA_DIR, "0"),
    os.path.join(DATA_DIR, "1"),
    os.path.join(DATA_DIR, "4")
]
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

# Evaluation dataset (dataset 5)
EVAL_IMAGES_DIR = os.path.join(DATA_DIR, "5")
EVAL_LABELS_CSV = os.path.join(DATA_DIR, "train_labels_5.csv")

# V5 specific paths
OUTPUT_DIR = os.path.join(V5_DIR, "outputs_v5")
MODELS_DIR = os.path.join(V5_DIR, "models_v5")
LOGS_DIR = os.path.join(V5_DIR, "logs_v5")
CHECKPOINTS_DIR = os.path.join(V5_DIR, "checkpoints_v5")
PREPROCESSED_DATA_DIR = os.path.join(V5_DIR, "preprocessed_data")
SAMPLES_OUTPUT_DIR = os.path.join(V5_DIR, "preprocessing_samples")

# Model parameters - IMPROVED from V4
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 30  # More epochs with better early stopping
LEARNING_RATE = 1e-4  # INCREASED: Better early exploration
MODEL_BACKBONE = "efficientnet-b0"

# Loss function - Weighted BCE (proven stable)
USE_FOCAL_LOSS = False

# Class weights - INCREASED for 29:1 imbalance
# V4 used {0: 1.0, 1: 5.0} -> validation recall dropped to 0.0
# V5 uses more aggressive weighting
CLASS_WEIGHTS = {
    0: 1.0,
    1: 10.0,  # INCREASED from 5.0 to 10.0 to force attention to minority class
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.199
TEST_SPLIT = 0.001
RANDOM_SEED = 42

# MODERATE augmentation (same as V4 - worked well)
AUGMENTATION = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'fill_mode': 'constant',
    'cval': 0
}

# V4 Advanced Preprocessing (KEEP - worked well)
USE_OD_DETECTION = True
OD_CROP_FACTOR = 3.0
OD_BRIGHTNESS_PERCENTILE = 99
OD_BRIGHTNESS_THRESHOLD = 0.9
OD_MIN_DIAMETER = 20
OD_MAX_DIAMETER = 300
OD_FALLBACK_TO_FULL = True  # Use full image if OD detection fails

# Advanced CLAHE - LAB color space (KEEP - working well)
USE_CLAHE = True
CLAHE_COLOR_SPACE = 'LAB'  # LAB, HSV, or RGB
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# Vessel enhancement (DISABLE - adds overhead without clear benefit)
USE_VESSEL_ENHANCEMENT = False
VESSEL_ALPHA = 0.3

# Callbacks - IMPROVED
EARLY_STOPPING_PATIENCE = 7  # INCREASED from 5 - give model more time
REDUCE_LR_PATIENCE = 3  # KEEP - worked in V4
REDUCE_LR_FACTOR = 0.5  # KEEP
MIN_LR = 1e-7

# Monitoring metric for callbacks
MONITOR_METRIC = 'val_auc'  # Focus on AUC, not loss
MONITOR_MODE = 'max'

# Mixed precision training
USE_MIXED_PRECISION = False

# Save sample preprocessed images for inspection
SAVE_PREPROCESSING_SAMPLES = True
NUM_SAMPLES_TO_SAVE = 10

# ========================
# V5 KEY IMPROVEMENTS
# ========================
"""
Based on V4 analysis (13 hours training, 12 epochs):

PROBLEMS IDENTIFIED:
1. Validation recall dropped to 0.0 (epochs 8-11) - too conservative
2. Class weight 5.0 insufficient for 29:1 imbalance
3. Learning rate 5e-5 too low - slow exploration

V5 FIXES:
1. CLASS_WEIGHTS[1] = 10.0 (doubled from 5.0)
   -> Forces model to pay more attention to minority class
   
2. LEARNING_RATE = 1e-4 (doubled from 5e-5)
   -> Faster initial learning, better exploration
   -> Will be reduced by ReduceLROnPlateau if needed
   
3. EARLY_STOPPING_PATIENCE = 7 (increased from 5)
   -> Give model more recovery time after val_recall=0 episodes
   
4. MONITOR_METRIC = 'val_auc' (not val_loss)
   -> AUC is more robust to class imbalance than raw loss

KEPT FROM V4 (working well):
- Optic disk detection + cropping
- Advanced CLAHE (LAB color space)
- Moderate augmentation
- Weighted BCE (not focal loss)
- ReduceLROnPlateau (helped at epoch 12)

EXPECTED IMPROVEMENTS:
- Validation recall should stay > 0.1 throughout training
- Better balance between precision and recall
- Faster convergence due to higher initial LR
- More stable validation metrics
"""
