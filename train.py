"""
Main training script for AIROGS Glaucoma Detection Baseline
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Import local modules
import config
from dataset import AIROGSDataset
from model import create_baseline_model, compile_model, get_callbacks
from evaluation import (
    evaluate_model,
    print_evaluation_results,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_prediction_distribution,
)


def setup_gpu():
    """Configure GPU settings"""
    import os

    # Display environment variables for debugging
    print("\n🔍 GPU Environment Check:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')}")

    # Check for GPU devices
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            print(f"\n✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                # Set memory growth to avoid OOM errors
                tf.config.experimental.set_memory_growth(gpu, True)

            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if gpu_details:
                print(f"   Device name: {gpu_details.get('device_name', 'Unknown')}")

            # Enable mixed precision for faster training
            if config.USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"\n⚡ Mixed precision enabled: {policy.name}")

            print(f"✅ GPU configuration successful!\n")

        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            print("⚠️  Falling back to CPU\n")
    else:
        print("\n❌ No GPU found!")
        print("⚠️  Training will run on CPU (very slow)")
        print("   Possible causes:")
        print("   - CUDA libraries not installed or not in PATH")
        print("   - TensorFlow not compiled with GPU support")
        print("   - GPU not allocated by SLURM (check --gres=gpu:1)")
        print("   - Wrong TensorFlow version for your CUDA version\n")


def train_model(args):
    """Main training function"""

    # Setup
    print("\n" + "=" * 60)
    print("AIROGS GLAUCOMA DETECTION - BASELINE TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {config.MODEL_BACKBONE}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("=" * 60 + "\n")

    # Configure GPU
    setup_gpu()

    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Load and prepare data
    print("\n📊 Loading dataset...")
    dataset = AIROGSDataset(
        labels_csv=config.TRAIN_LABELS_CSV, images_dir=config.TRAIN_IMAGES_DIR
    )

    train_df, val_df, test_df = dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED,
    )

    # Create data generators
    print("\n🔄 Creating data generators...")
    train_gen, val_gen, test_gen = dataset.create_generators(
        batch_size=config.BATCH_SIZE, augment=True
    )

    # Calculate steps per epoch
    steps_per_epoch = len(train_df) // config.BATCH_SIZE
    validation_steps = len(val_df) // config.BATCH_SIZE

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Create model
    print("\n🏗️  Building model...")
    model = create_baseline_model(
        backbone=config.MODEL_BACKBONE,
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=1,
    )

    # Compile model
    model = compile_model(
        model, learning_rate=config.LEARNING_RATE, class_weights=config.CLASS_WEIGHTS
    )

    # Print model summary
    model.summary()

    # Get callbacks
    model_name = f"airogs_baseline_{config.MODEL_BACKBONE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    callbacks = get_callbacks(
        model_name=model_name, patience=config.EARLY_STOPPING_PATIENCE
    )

    # Train model
    print("\n🚀 Starting training...")
    print(f"Training on {len(train_df)} images")
    print(f"Validating on {len(val_df)} images")

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

    # Save training history
    history_path = os.path.join(config.LOGS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        # Convert numpy types to native Python types
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")

    return model, history


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train AIROGS glaucoma detection baseline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (overrides config)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="Image size (overrides config)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["efficientnet-b0", "resnet50", "efficientnet-b3"],
        help="Model backbone (overrides config)",
    )

    args = parser.parse_args()

    # Override config if arguments provided
    if args.data_dir:
        config.DATA_DIR = args.data_dir
        config.TRAIN_IMAGES_DIR = os.path.join(config.DATA_DIR, "0")
        config.TRAIN_LABELS_CSV = os.path.join(config.DATA_DIR, "train_labels.csv")
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.image_size:
        config.IMAGE_SIZE = args.image_size
    if args.backbone:
        config.MODEL_BACKBONE = args.backbone

    # Train model
    model, history = train_model(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
