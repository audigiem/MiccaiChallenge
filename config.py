"""
Configuration file for AIROGS Glaucoma Detection Baseline
"""

import os

# Paths
DATA_DIR = "./dataset/"
# Use datasets 0, 1, and 4 for training
TRAIN_IMAGES_DIR = [
    os.path.join(DATA_DIR, "0"),
    os.path.join(DATA_DIR, "1"),
    os.path.join(DATA_DIR, "4")
]
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
OUTPUT_DIR = "./outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Model parameters
IMAGE_SIZE = 384  # Resize images to 384x384 for faster training
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 5e-5  # Reduced from 1e-4 to prevent overfitting
MODEL_BACKBONE = "efficientnet-b0"  # Options: "efficientnet-b0", "resnet50"

# Class imbalance handling
# For datasets 0, 1, 4: typical imbalance ratio ~1:30
# Using moderate weight to avoid extreme overfitting on minority class
CLASS_WEIGHTS = {
    0: 1.0,  # NRG (No Referable Glaucoma)
    1: 5.0,  # RG (Referable Glaucoma) - reduced from 10 to prevent overfitting
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.199
TEST_SPLIT = 0.001  # No separate test set in this baseline
RANDOM_SEED = 42

# Augmentation parameters (RE-ENABLED to prevent overfitting)
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,  # Images can be rotated
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2],
}

# Training parameters
EARLY_STOPPING_PATIENCE = 8  # Increased to allow more exploration
REDUCE_LR_PATIENCE = 4  # Increased from 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7  # Minimum learning rate

# Evaluation
SPECIFICITY_THRESHOLD = 0.95  # For sensitivity @ 95% specificity
PAUC_RANGE = (0.9, 1.0)  # Partial AUC range (90-100% specificity)

# GPU settings
USE_MIXED_PRECISION = False  # Disabled due to cuDNN issues with V100
