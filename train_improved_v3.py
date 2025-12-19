"""
Week 2 Training Script V3 - BACK TO STABLE BASELINE
Learning from V2 failures:
- V2 used focal loss -> unstable predictions
- V2 had unfreezing configs but no implementation -> confusion
- V2 was too aggressive with changes

V3 Strategy:
- Use proven weighted BCE (not focal loss)
- Keep CLAHE (only proven improvement)
- Moderate augmentation
- Stable training (no fancy tricks)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Import local modules
import config_improved_v3 as config
from dataset import AIROGSDataset
from model import create_baseline_model, get_callbacks
from improvements_week2 import apply_clahe_preprocessing


def setup_gpu():
    """Configure GPU settings"""
    print("\n🔍 GPU Environment Check:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    gpus = tf.config.list_physical_devices("GPU")
    
    if gpus:
        try:
            print(f"\n✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                tf.config.experimental.set_memory_growth(gpu, True)
            
            if config.USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"\n⚡ Mixed precision enabled: {policy.name}")
            
            print(f"✅ GPU configuration successful!\n")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("\n❌ No GPU found - training will be slow\n")


class CLAHEImageDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    """Custom ImageDataGenerator with CLAHE preprocessing"""
    
    def __init__(self, use_clahe=False, clahe_clip=2.0, clahe_tile=8, **kwargs):
        super().__init__(**kwargs)
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
    
    def standardize(self, x):
        """Apply CLAHE before standard preprocessing"""
        if self.use_clahe:
            # Apply CLAHE (x is in [0, 255] at this point)
            x = apply_clahe_preprocessing(x.astype(np.uint8), 
                                         clip_limit=self.clahe_clip,
                                         tile_size=self.clahe_tile)
            x = x * 255.0  # Convert back to [0, 255] for rescaling
        
        return super().standardize(x)


def create_improved_generators(dataset, batch_size=32):
    """Create data generators with CLAHE preprocessing"""
    
    if dataset.train_df is None:
        dataset.split_data()
    
    train_df = dataset.train_df
    val_df = dataset.val_df
    test_df = dataset.test_df
    
    # Training generator with augmentation and CLAHE
    train_datagen = CLAHEImageDataGenerator(
        use_clahe=config.USE_CLAHE,
        clahe_clip=config.CLAHE_CLIP_LIMIT,
        clahe_tile=config.CLAHE_TILE_SIZE,
        rescale=1.0 / 255,
        horizontal_flip=config.AUGMENTATION.get("horizontal_flip", True),
        vertical_flip=config.AUGMENTATION.get("vertical_flip", True),
        rotation_range=config.AUGMENTATION.get("rotation_range", 15),
        width_shift_range=config.AUGMENTATION.get("width_shift_range", 0.1),
        height_shift_range=config.AUGMENTATION.get("height_shift_range", 0.1),
        zoom_range=config.AUGMENTATION.get("zoom_range", 0.1),
        brightness_range=config.AUGMENTATION.get("brightness_range", [0.8, 1.2]),
        fill_mode="constant",
        cval=0,
    )
    
    # Validation generator with CLAHE but no augmentation
    val_datagen = CLAHEImageDataGenerator(
        use_clahe=config.USE_CLAHE,
        clahe_clip=config.CLAHE_CLIP_LIMIT,
        clahe_tile=config.CLAHE_TILE_SIZE,
        rescale=1.0 / 255
    )
    
    # Test generator
    test_datagen = CLAHEImageDataGenerator(
        use_clahe=config.USE_CLAHE,
        clahe_clip=config.CLAHE_CLIP_LIMIT,
        clahe_tile=config.CLAHE_TILE_SIZE,
        rescale=1.0 / 255
    )
    
    # Create generators
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="image_path",
        y_col="label",
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="image_path",
        y_col="label",
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col="label",
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen


def train_model():
    """Main training function - V3 STABLE"""
    
    # Setup
    print("\n" + "=" * 70)
    print("AIROGS GLAUCOMA DETECTION - V3 STABLE TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🔄 V3 CHANGES FROM V2:")
    print("   ✅ BACK TO Weighted BCE (focal loss was unstable)")
    print("   ✅ Conservative LR: 5e-5 (baseline worked)")
    print("   ✅ No progressive unfreezing (kept it simple)")
    print("   ✅ 20 epochs (moderate)")
    print("   ✅ KEEPING: CLAHE preprocessing + moderate augmentation")
    
    setup_gpu()
    
    # Configuration summary
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.MODEL_BACKBONE}")
    print(f"   Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Loss: Weighted BCE (weights={config.CLASS_WEIGHTS})")
    print(f"   CLAHE preprocessing: {config.USE_CLAHE}")
    
    # Create output directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
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
    
    # Create data generators
    print(f"\n🔄 Creating data generators...")
    train_gen, val_gen, test_gen = create_improved_generators(
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
    
    # Compile model with WEIGHTED BCE (not focal loss!)
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
    model_name = f"airogs_improved_v3_{config.MODEL_BACKBONE}_{timestamp}"
    
    callbacks = get_callbacks(
        model_name=model_name,
        patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Train model
    print(f"\n🚀 Starting training...")
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
        "backbone": config.MODEL_BACKBONE,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "loss": "weighted_bce",
        "class_weights": config.CLASS_WEIGHTS,
        "clahe": config.USE_CLAHE,
        "augmentation": config.AUGMENTATION,
        "timestamp": timestamp
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest model: {os.path.join(config.MODELS_DIR, f'{model_name}_best.keras')}")
    print(f"Final model: {final_model_path}")
    print(f"Training log: {os.path.join(config.LOGS_DIR, f'{model_name}_training.csv')}")
    
    return model, history, final_model_path


if __name__ == "__main__":
    try:
        model, history, model_path = train_model()
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("\n1. Evaluate with TTA:") 
        print(f"   python evaluation_improved.py {model_path} --tta")
        print("\n2. Evaluate with TTA + CLAHE:")
        print(f"   python evaluation_improved.py {model_path} --tta --clahe")
        print("\n" + "=" * 70)
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
