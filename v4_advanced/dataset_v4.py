"""
V4 Dataset Handler with Advanced Preprocessing
Integrates optic disk detection and advanced CLAHE
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from dataset import AIROGSDataset

# Import advanced preprocessing
from advanced_preprocessing import (
    preprocess_fundus_advanced,
    detect_optic_disk_simple,
    crop_around_optic_disk,
    apply_advanced_clahe,
)

import config_v4 as config


class V4AdvancedImageDataGenerator(ImageDataGenerator):
    """
    Custom ImageDataGenerator with V4 advanced preprocessing:
    - Optic disk detection & cropping
    - Advanced CLAHE in LAB color space
    - Optional vessel enhancement
    """

    def __init__(
        self,
        use_od_crop=True,
        use_clahe=True,
        use_vessel=False,
        clahe_clip=2.0,
        clahe_tile=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_od_crop = use_od_crop
        self.use_clahe = use_clahe
        self.use_vessel = use_vessel
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.preprocessing_stats = {
            "od_detected": 0,
            "od_failed": 0,
            "total_processed": 0,
        }

    def standardize(self, x):
        """
        Apply V4 advanced preprocessing before standard normalization
        x is in [0, 255] at this point
        """
        # Convert to uint8
        x = x.astype(np.uint8)

        # 1. Optic Disk Detection & Cropping
        if self.use_od_crop:
            try:
                od_info = detect_optic_disk_simple(x)
                if od_info is not None:
                    x = crop_around_optic_disk(
                        x, od_info, crop_factor=config.OD_CROP_FACTOR
                    )
                    # Resize back to target size after cropping
                    x = cv2.resize(
                        x,
                        (config.IMAGE_SIZE, config.IMAGE_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                    self.preprocessing_stats["od_detected"] += 1
                else:
                    self.preprocessing_stats["od_failed"] += 1
                    if not config.OD_FALLBACK_TO_FULL:
                        # If we require OD and it failed, apply heavy preprocessing
                        pass
            except Exception as e:
                # If OD detection fails, continue with full image
                self.preprocessing_stats["od_failed"] += 1

        # 2. Advanced CLAHE
        if self.use_clahe:
            try:
                x = apply_advanced_clahe(
                    x,
                    clip_limit=self.clahe_clip,
                    tile_size=self.clahe_tile,
                    color_mode=config.CLAHE_COLOR_SPACE,
                )
            except Exception as e:
                # Fallback to no CLAHE if fails
                print(f"Warning: CLAHE failed: {e}")

        # 3. Vessel Enhancement (optional - experimental)
        if self.use_vessel:
            try:
                from advanced_preprocessing import (
                    enhance_vessels,
                    extract_green_channel,
                )

                green_channel = extract_green_channel(x)
                vessels = enhance_vessels(green_channel)
                # Blend with original green channel
                x[:, :, 1] = (x[:, :, 1] * 0.7 + vessels * 0.3).astype(np.uint8)
            except Exception as e:
                print(f"Warning: Vessel enhancement failed: {e}")

        self.preprocessing_stats["total_processed"] += 1

        # Convert back to float for rescaling
        x = x.astype(np.float32)

        # Apply standard preprocessing (rescaling will be done by parent class)
        return super().standardize(x)

    def print_stats(self):
        """Print preprocessing statistics"""
        stats = self.preprocessing_stats
        total = stats["total_processed"]
        if total > 0:
            print(f"\n📊 V4 Preprocessing Statistics:")
            print(f"   Total processed: {total}")
            print(
                f"   OD detected: {stats['od_detected']} ({stats['od_detected']/total*100:.1f}%)"
            )
            print(
                f"   OD failed: {stats['od_failed']} ({stats['od_failed']/total*100:.1f}%)"
            )


def create_v4_generators(dataset, batch_size=32):
    """
    Create V4 data generators with advanced preprocessing

    Args:
        dataset: AIROGSDataset instance (already loaded and split)
        batch_size: Batch size for training

    Returns:
        train_gen, val_gen, test_gen with V4 preprocessing
    """

    if dataset.train_df is None:
        raise ValueError("Dataset must be split before creating generators")

    train_df = dataset.train_df
    val_df = dataset.val_df
    test_df = dataset.test_df

    # Training generator with V4 preprocessing + augmentation
    train_datagen = V4AdvancedImageDataGenerator(
        use_od_crop=config.USE_OD_DETECTION,
        use_clahe=config.USE_CLAHE,
        use_vessel=config.USE_VESSEL_ENHANCEMENT,
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

    # Validation generator with V4 preprocessing but no augmentation
    val_datagen = V4AdvancedImageDataGenerator(
        use_od_crop=config.USE_OD_DETECTION,
        use_clahe=config.USE_CLAHE,
        use_vessel=config.USE_VESSEL_ENHANCEMENT,
        clahe_clip=config.CLAHE_CLIP_LIMIT,
        clahe_tile=config.CLAHE_TILE_SIZE,
        rescale=1.0 / 255,
    )

    # Test generator
    test_datagen = V4AdvancedImageDataGenerator(
        use_od_crop=config.USE_OD_DETECTION,
        use_clahe=config.USE_CLAHE,
        use_vessel=config.USE_VESSEL_ENHANCEMENT,
        clahe_clip=config.CLAHE_CLIP_LIMIT,
        clahe_tile=config.CLAHE_TILE_SIZE,
        rescale=1.0 / 255,
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
        seed=config.RANDOM_SEED,
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="image_path",
        y_col="label",
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col="label",
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen, train_datagen


def save_preprocessing_samples(dataset, num_samples=10):
    """
    Save sample preprocessed images for visual inspection

    Args:
        dataset: AIROGSDataset instance
        num_samples: Number of samples to save
    """
    import matplotlib.pyplot as plt

    os.makedirs(config.SAMPLES_OUTPUT_DIR, exist_ok=True)

    # Get random samples from training set
    sample_indices = np.random.choice(
        len(dataset.train_df), min(num_samples, len(dataset.train_df)), replace=False
    )

    for idx, sample_idx in enumerate(sample_indices):
        row = dataset.train_df.iloc[sample_idx]
        image_path = row["image_path"]
        label = row["label"]

        # Load original image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        # Apply V4 preprocessing
        processed = preprocess_fundus_advanced(
            image_path,
            target_size=config.IMAGE_SIZE,
            use_od_crop=config.USE_OD_DETECTION,
            use_clahe=config.USE_CLAHE,
            use_vessel_enhance=config.USE_VESSEL_ENHANCEMENT,
        )

        # Convert processed back to uint8 for visualization
        processed_vis = (processed * 255).astype(np.uint8)

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(original)
        axes[0].set_title(f"Original (Label: {label})")
        axes[0].axis("off")

        axes[1].imshow(processed_vis)
        axes[1].set_title(
            f"V4 Preprocessed (OD={config.USE_OD_DETECTION}, CLAHE={config.USE_CLAHE})"
        )
        axes[1].axis("off")

        plt.tight_layout()
        output_path = os.path.join(
            config.SAMPLES_OUTPUT_DIR, f"sample_{idx+1}_{os.path.basename(image_path)}"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"\n✅ Saved {num_samples} preprocessing samples to: {config.SAMPLES_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    # Test V4 preprocessing
    print("Testing V4 Advanced Preprocessing Pipeline...")

    # Load dataset
    dataset = AIROGSDataset(
        labels_csv=config.TRAIN_LABELS_CSV, images_dir=config.TRAIN_IMAGES_DIR
    )
    dataset.load_data()
    dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED,
    )

    print(f"\n✅ Dataset loaded: {len(dataset.df)} images")

    # Create generators
    print("\n🔄 Creating V4 generators...")
    train_gen, val_gen, test_gen, train_datagen = create_v4_generators(
        dataset, batch_size=config.BATCH_SIZE
    )

    print(f"✅ Generators created")
    print(f"   Training steps: {len(train_gen)}")
    print(f"   Validation steps: {len(val_gen)}")

    # Save preprocessing samples
    if config.SAVE_PREPROCESSING_SAMPLES:
        print("\n📸 Generating preprocessing samples...")
        save_preprocessing_samples(dataset, num_samples=config.NUM_SAMPLES_TO_SAVE)

    print("\n✅ V4 preprocessing test complete!")
