"""
Configuration for Week 2 - V3 STABLE VERSION
Goes back to what worked, with careful improvements
"""

import os

# Paths
DATA_DIR = "./dataset/"
TRAIN_IMAGES_DIR = [
    os.path.join(DATA_DIR, "0"),
    os.path.join(DATA_DIR, "1"),
    os.path.join(DATA_DIR, "4")
]
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
OUTPUT_DIR = "./outputs_improved"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Model parameters - OPTIMIZED from V3 results analysis
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 25  # Increased: V3 stopped at epoch 9 due to walltime
LEARNING_RATE = 7.5e-5  # INCREASED: 1.5x baseline for faster convergence
MODEL_BACKBONE = "efficientnet-b0"

# BACK TO WEIGHTED BCE - Focal loss was too unstable
USE_FOCAL_LOSS = False
FOCAL_LOSS_GAMMA = 2.0  # Not used, keeping for compatibility
FOCAL_LOSS_ALPHA = 0.25

# Class weights - INCREASED (V3 had val_recall instability: 0.39→0.09)
CLASS_WEIGHTS = {
    0: 1.0,
    1: 8.0,  # INCREASED from 5.0 to address val_recall drops
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.199
TEST_SPLIT = 0.001
RANDOM_SEED = 42

# MODERATE augmentation - not too aggressive
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2],
}

# CLAHE preprocessing - KEEP THIS
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# Training parameters - IMPROVED from V3 analysis
EARLY_STOPPING_PATIENCE = 6  # Reduced: V3 only needed 9 epochs
REDUCE_LR_PATIENCE = 3  # More responsive to plateaus
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# DISABLE advanced features that caused instability
USE_LR_WARMUP = False
UNFREEZE_AT_EPOCH = 0  # No unfreezing

# Evaluation
SPECIFICITY_THRESHOLD = 0.95
PAUC_RANGE = (0.9, 1.0)

# Test-time augmentation
USE_TTA = True
TTA_AUGMENTATIONS = 5

# GPU settings
USE_MIXED_PRECISION = False
