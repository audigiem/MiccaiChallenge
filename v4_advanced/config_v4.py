"""
Configuration for V4 - ADVANCED PREPROCESSING VERSION
Includes optic disk detection, advanced CLAHE, and proven training methods
"""

import os

# Paths
DATA_DIR = "dataset/"
TRAIN_IMAGES_DIR = [
    os.path.join(DATA_DIR, "0"),
    os.path.join(DATA_DIR, "1"),
    os.path.join(DATA_DIR, "4")
]
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

# V4 specific paths
V4_DIR = "./"  # v4_advanced directory
OUTPUT_DIR = os.path.join(V4_DIR, "outputs")
MODELS_DIR = os.path.join(V4_DIR, "models")
LOGS_DIR = os.path.join(V4_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(V4_DIR, "checkpoints")
PREPROCESSED_DATA_DIR = os.path.join(V4_DIR, "preprocessed_data")

# Model parameters - Conservative and proven (from V3)
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 25  # Slightly more than V3 for advanced preprocessing
LEARNING_RATE = 5e-5  # Proven baseline value
MODEL_BACKBONE = "efficientnet-b0"

# Loss function - Weighted BCE (proven stable)
USE_FOCAL_LOSS = False
FOCAL_LOSS_GAMMA = 2.0  # Not used
FOCAL_LOSS_ALPHA = 0.25  # Not used

# Class weights - PROVEN to work
CLASS_WEIGHTS = {
    0: 1.0,
    1: 5.0,  # Baseline ratio
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.199
TEST_SPLIT = 0.001
RANDOM_SEED = 42

# MODERATE augmentation
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2],
}

# ========================================
# V4 ADVANCED PREPROCESSING FEATURES
# ========================================

# Optic Disk Detection & Cropping
USE_OD_DETECTION = True
OD_CROP_FACTOR = 3.0  # How much larger than OD to crop (3x radius)
OD_FALLBACK_TO_FULL = True  # Use full image if OD not detected

# Advanced CLAHE
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
CLAHE_COLOR_SPACE = 'LAB'  # 'LAB', 'HSV', or 'RGB'

# Vessel Enhancement (Experimental - can disable)
USE_VESSEL_ENHANCEMENT = False  # Start False, can enable after testing

# Preprocessing mode
PREPROCESS_MODE = 'on_the_fly'  # 'on_the_fly' or 'precomputed'
# 'on_the_fly': Apply preprocessing during training (slower but flexible)
# 'precomputed': Preprocess entire dataset once and save (faster training)

# ========================================
# TRAINING PARAMETERS
# ========================================

EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Disable advanced features for stability
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

# ========================================
# PREPROCESSING VISUALIZATION
# ========================================

# Save sample preprocessed images for inspection
SAVE_PREPROCESSING_SAMPLES = True
NUM_SAMPLES_TO_SAVE = 10
SAMPLES_OUTPUT_DIR = os.path.join(V4_DIR, "preprocessing_samples")
