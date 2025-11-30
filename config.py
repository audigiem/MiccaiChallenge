"""
Configuration file for AIROGS Glaucoma Detection Baseline
"""
import os

# Paths
DATA_DIR = "./data"
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "0")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
OUTPUT_DIR = "./outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Model parameters
IMAGE_SIZE = 384  # Resize images to 384x384 for faster training
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_BACKBONE = "efficientnet-b0"  # Options: "efficientnet-b0", "resnet50"

# Class imbalance handling
# RG: 3270, NRG: 98172 -> imbalance ratio ~1:30
CLASS_WEIGHTS = {
    0: 1.0,    # NRG (No Referable Glaucoma)
    1: 30.0    # RG (Referable Glaucoma)
}

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Augmentation parameters
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2],
}

# Training parameters
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# Evaluation
SPECIFICITY_THRESHOLD = 0.95  # For sensitivity @ 95% specificity
PAUC_RANGE = (0.9, 1.0)  # Partial AUC range (90-100% specificity)

# GPU settings
USE_MIXED_PRECISION = True

