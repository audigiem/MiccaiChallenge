"""
Configuration for Week 2 improved training
Uses focal loss and other data-focused improvements
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
BATCH_SIZE = 16  # Reduced from 32 for RTX 6000 stability
EPOCHS = 30
LEARNING_RATE = 5e-5
MODEL_BACKBONE = "efficientnet-b0"

# Use FOCAL LOSS instead of weighted BCE
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

# Class weights (only used if USE_FOCAL_LOSS = False)
CLASS_WEIGHTS = {
    0: 1.0,
    1: 5.0,
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.199
TEST_SPLIT = 0.001
RANDOM_SEED = 42

# Enhanced augmentation parameters
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 20,  # Increased from 15
    "width_shift_range": 0.15,  # Increased from 0.1
    "height_shift_range": 0.15,  # Increased from 0.1
    "zoom_range": 0.15,  # Increased from 0.1
    "brightness_range": [0.7, 1.3],  # Wider range
}

# Use CLAHE preprocessing
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# Training parameters
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Evaluation
SPECIFICITY_THRESHOLD = 0.95
PAUC_RANGE = (0.9, 1.0)

# Test-time augmentation
USE_TTA = True
TTA_AUGMENTATIONS = 5

# GPU settings - RTX 6000 compatible
USE_MIXED_PRECISION = False  # Disabled for RTX 6000 stability
