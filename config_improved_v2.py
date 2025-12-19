"""
Configuration for Week 2 - OPTIMIZED VERSION
Fixes issues from previous training attempt
"""

import os

# Paths (same as baseline)
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

# Model parameters - Optimized for RTX 6000 (24GB VRAM)
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 15  # Reduced from 30 - your model learns fast after epoch 14
LEARNING_RATE = 1e-4  # Increased from 5e-5 for faster initial learning
MODEL_BACKBONE = "efficientnet-b0"

# OPTIMIZED FOCAL LOSS PARAMETERS
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 1.5  # Reduced from 2.0 - less aggressive
FOCAL_LOSS_ALPHA = 0.75  # Increased from 0.25 - more weight on positive class

# Class weights (only used if USE_FOCAL_LOSS = False)
CLASS_WEIGHTS = {
    0: 1.0,
    1: 8.0,  # Reduced from 10.0, more balanced
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.199
TEST_SPLIT = 0.001
RANDOM_SEED = 42

# Moderate augmentation (your previous was too aggressive)
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 15,  # Reduced from 20
    "width_shift_range": 0.1,  # Reduced from 0.15
    "height_shift_range": 0.1,  # Reduced from 0.15
    "zoom_range": 0.1,  # Reduced from 0.15
    "brightness_range": [0.8, 1.2],  # Narrower range than [0.7, 1.3]
}

# CLAHE preprocessing - KEEP THIS, it's good
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# IMPROVED TRAINING PARAMETERS
EARLY_STOPPING_PATIENCE = 6  # Reduced from 8
REDUCE_LR_PATIENCE = 3  # Reduced from 4 for faster adaptation
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# LEARNING RATE WARMUP (NEW!)
USE_LR_WARMUP = True
WARMUP_EPOCHS = 2  # Gradual warmup to prevent initial instability

# PROGRESSIVE UNFREEZING (NEW!)
UNFREEZE_AT_EPOCH = 5  # Start fine-tuning backbone after 5 epochs
UNFREEZE_LAYERS = 50  # Number of layers to unfreeze (from the end)

# Evaluation
SPECIFICITY_THRESHOLD = 0.95
PAUC_RANGE = (0.9, 1.0)

# Test-time augmentation
USE_TTA = True
TTA_AUGMENTATIONS = 5

# GPU settings - RTX 6000 compatible
USE_MIXED_PRECISION = False  # Keep disabled for stability
