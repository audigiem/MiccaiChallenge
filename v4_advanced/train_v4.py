"""
V4 Training Script - ADVANCED PREPROCESSING
Integrates optic disk detection, advanced CLAHE, and proven training methods
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Add current directory and v4_advanced to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import local modules
import config_v4 as config
from dataset_v4 import create_v4_generators, save_preprocessing_samples
from dataset import AIROGSDataset
from model import create_baseline_model, get_callbacks


def setup_gpu():
    """Configure GPU settings"""
    print("\n🔍 GPU Environment Check:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Set memory growth before listing devices
    gpus = tf.config.list_physical_devices("GPU")
    
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Now list logical devices to verify
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"\n✅ Found {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            if config.USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"\n⚡ Mixed precision enabled: {policy.name}")
            
            print(f"\n✅ GPU configuration successful!\n")
            return True
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("\n⚠️  No GPU found - training will be VERY slow\n")
        print("Possible issues:")
        print("  - TensorFlow not built with GPU support")
        print("  - CUDA/cuDNN version mismatch")
        print("  - GPU driver issues")
        return False


def train_v4_model():
    """Main V4 training function with advanced preprocessing"""
    
    # Setup
    print("\n" + "=" * 70)
    print("AIROGS GLAUCOMA DETECTION - V4 ADVANCED PREPROCESSING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🚀 V4 NEW FEATURES:")
    print(f"   ✨ Optic Disk Detection & Cropping: {config.USE_OD_DETECTION}")
    print(f"   ✨ Advanced CLAHE ({config.CLAHE_COLOR_SPACE} color space): {config.USE_CLAHE}")
    print(f"   ✨ Vessel Enhancement: {config.USE_VESSEL_ENHANCEMENT}")
    print("\n🔄 PROVEN METHODS FROM V3:")
    print("   ✅ Weighted BCE (stable)")
    print("   ✅ Conservative LR: 5e-5")
    print("   ✅ Moderate augmentation")
    
    setup_gpu()
    
    # Configuration summary
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.MODEL_BACKBONE}")
    print(f"   Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Loss: Weighted BCE (weights={config.CLASS_WEIGHTS})")
    print(f"   OD crop factor: {config.OD_CROP_FACTOR}x")
    
    # Create output directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    print(f"   Data directories: {config.TRAIN_IMAGES_DIR}")
    print(f"   Labels CSV: {config.TRAIN_LABELS_CSV}")
    
    dataset = AIROGSDataset(
        labels_csv=config.TRAIN_LABELS_CSV,
        images_dir=config.TRAIN_IMAGES_DIR
    )
    dataset.load_data()
    
    train_df, val_df, test_df = dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total samples: {len(dataset.df)}")
    print(f"   Training: {len(dataset.train_df)}")
    print(f"   Validation: {len(dataset.val_df)}")
    print(f"   Test: {len(dataset.test_df)}")
    
    # Class distribution
    train_counts = dataset.train_df["label"].value_counts()
    print(f"\n   Training class distribution:")
    print(f"      NRG (0): {train_counts.get(0, 0)}")
    print(f"      RG (1):  {train_counts.get(1, 0)}")
    print(f"      Imbalance ratio: {train_counts.get(0, 0) / max(train_counts.get(1, 1), 1):.1f}:1")
    
    # Save preprocessing samples for inspection
    if config.SAVE_PREPROCESSING_SAMPLES:
        print(f"\n📸 Saving preprocessing samples...")
        save_preprocessing_samples(dataset, num_samples=config.NUM_SAMPLES_TO_SAVE)
    
    # Create V4 data generators with advanced preprocessing
    print(f"\n🔄 Creating V4 advanced data generators...")
    train_gen, val_gen, test_gen, train_datagen = create_v4_generators(
        dataset, batch_size=config.BATCH_SIZE
    )
    
    steps_per_epoch = len(train_df) // config.BATCH_SIZE
    validation_steps = len(val_df) // config.BATCH_SIZE
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Create model
    print(f"\n🏗️  Building model...")
    model = create_baseline_model(
        backbone=config.MODEL_BACKBONE,
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=1
    )
    
    print(f"\n   Model summary:")
    print(f"   Total params: {model.count_params():,}")
    
    # Compile model with WEIGHTED BCE
    print(f"\n⚙️  Compiling model...")
    print(f"   Using Weighted Binary Crossentropy (STABLE)")
    print(f"   Class weights: {config.CLASS_WEIGHTS}")
    
    def weighted_bce(y_true, y_pred):
        weights = tf.where(tf.equal(y_true, 1), 
                         config.CLASS_WEIGHTS[1], 
                         config.CLASS_WEIGHTS[0])
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * bce)
    
    loss_fn = weighted_bce
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"airogs_v4_advanced_{config.MODEL_BACKBONE}_{timestamp}"
    
    callbacks = get_callbacks(
        model_name=model_name,
        patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Train model
    print(f"\n🚀 Starting V4 training with advanced preprocessing...")
    print(f"   Model name: {model_name}")
    print(f"   Training on {len(train_df)} images")
    print(f"   Validating on {len(val_df)} images")
    print(f"   {'='*70}\n")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print preprocessing statistics
    print("\n" + "=" * 70)
    train_datagen.print_stats()
    print("=" * 70)
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, f"{model_name}_final.keras")
    model.save(final_model_path, save_format="keras")
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(config.LOGS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save training config
    config_path = os.path.join(config.MODELS_DIR, f"{model_name}_config.json")
    config_dict = {
        "model_name": model_name,
        "version": "v4_advanced",
        "backbone": config.MODEL_BACKBONE,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "loss": "weighted_bce",
        "class_weights": config.CLASS_WEIGHTS,
        "preprocessing": {
            "optic_disk_detection": config.USE_OD_DETECTION,
            "od_crop_factor": config.OD_CROP_FACTOR,
            "clahe": config.USE_CLAHE,
            "clahe_color_space": config.CLAHE_COLOR_SPACE,
            "vessel_enhancement": config.USE_VESSEL_ENHANCEMENT
        },
        "augmentation": config.AUGMENTATION,
        "timestamp": timestamp
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ V4 TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest model: {os.path.join(config.MODELS_DIR, f'{model_name}_best.keras')}")
    print(f"Final model: {final_model_path}")
    print(f"Training log: {os.path.join(config.LOGS_DIR, f'{model_name}_training.csv')}")
    print(f"Preprocessing samples: {config.SAMPLES_OUTPUT_DIR}")
    
    return model, history, final_model_path


if __name__ == "__main__":
    try:
        model, history, model_path = train_v4_model()
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("\n1. Inspect preprocessing samples:")
        print(f"   Check images in: {config.SAMPLES_OUTPUT_DIR}")
        print("\n2. Evaluate with TTA:")
        print(f"   python ../evaluation_improved.py {model_path} --tta")
        print("\n3. Evaluate with TTA + CLAHE:")
        print(f"   python ../evaluation_improved.py {model_path} --tta --clahe")
        print("\n" + "=" * 70)
    except Exception as e:
        print(f"\n❌ V4 Training failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
