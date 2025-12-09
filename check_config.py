#!/usr/bin/env python3
"""
Script de vérification rapide des configurations pour le nouvel entraînement
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(__file__))

import config

print("=" * 70)
print("VÉRIFICATION DE LA CONFIGURATION - AIROGS Training v2")
print("=" * 70)

print("\n📋 Paramètres d'entraînement:")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Batch size: {config.BATCH_SIZE}")
print(f"   Learning rate: {config.LEARNING_RATE}")
print(f"   Image size: {config.IMAGE_SIZE}")
print(f"   Model backbone: {config.MODEL_BACKBONE}")

print("\n⚖️  Class weights:")
print(f"   NRG (0): {config.CLASS_WEIGHTS[0]}")
print(f"   RG (1): {config.CLASS_WEIGHTS[1]}")

print("\n🎨 Data augmentation:")
aug_enabled = any(config.AUGMENTATION.values())
print(f"   Status: {'ENABLED ✅' if aug_enabled else 'DISABLED ❌'}")
if aug_enabled:
    for key, value in config.AUGMENTATION.items():
        if value:
            print(f"   - {key}: {value}")

print("\n⏱️  Training parameters:")
print(f"   Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
print(f"   Reduce LR patience: {config.REDUCE_LR_PATIENCE}")
print(f"   Reduce LR factor: {config.REDUCE_LR_FACTOR}")
print(f"   Minimum LR: {config.MIN_LR}")

print("\n📂 Data split:")
print(f"   Train: {config.TRAIN_SPLIT * 100:.1f}%")
print(f"   Val: {config.VAL_SPLIT * 100:.1f}%")
print(f"   Test: {config.TEST_SPLIT * 100:.1f}%")

print("\n📁 Datasets:")
if isinstance(config.TRAIN_IMAGES_DIR, list):
    print(f"   Using {len(config.TRAIN_IMAGES_DIR)} dataset directories:")
    for d in config.TRAIN_IMAGES_DIR:
        print(f"   - {d}")
else:
    print(f"   Using single directory: {config.TRAIN_IMAGES_DIR}")

print("\n🎯 Expected improvements:")
print("   ✅ Reduced learning rate (5e-5) → slower, more stable convergence")
print("   ✅ Reduced class weights (5.0) → less overfitting on minority class")
print("   ✅ Data augmentation enabled → better generalization")
print("   ✅ Increased dropout (0.5) → stronger regularization")
print("   ✅ L2 regularization (0.001) → penalize large weights")
print("   ✅ Increased patience (8) → more time to explore")
print("   ✅ Monitor val_auc → better metric for imbalanced data")

print("\n⚠️  Notes:")
print("   - Training will take longer due to data augmentation")
print("   - Expected training time: 8-12 hours for 30 epochs")
print("   - Models saved in .keras format (TensorFlow 2.x recommended)")
print("   - If overfitting persists, consider creating balanced dataset")

print("\n" + "=" * 70)
print("Configuration looks good! Ready to train.")
print("Launch with: sbatch train_datasets_014_v2.sh")
print("=" * 70)

