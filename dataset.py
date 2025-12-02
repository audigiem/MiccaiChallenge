"""
Dataset and data loading utilities for AIROGS challenge
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


class AIROGSDataset:
    """Handle AIROGS dataset loading and preprocessing"""

    def __init__(self, labels_csv, images_dir):
        self.labels_csv = labels_csv
        self.images_dir = images_dir
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data(self):
        """Load and preprocess the CSV labels"""
        self.df = pd.read_csv(self.labels_csv)

        # Convert labels to binary: RG=1, NRG=0
        self.df["label"] = (self.df["class"] == "RG").astype(int)

        # Add full image paths
        self.df["image_path"] = self.df["challenge_id"].apply(
            lambda x: os.path.join(self.images_dir, f"{x}.jpg")
        )

        # Check for missing files (optional, can be slow)
        # self.df = self.df[self.df['image_path'].apply(os.path.exists)]

        print(f"Total images: {len(self.df)}")
        print(f"RG (Glaucoma): {(self.df['label'] == 1).sum()}")
        print(f"NRG (No Glaucoma): {(self.df['label'] == 0).sum()}")
        print(
            f"Class imbalance ratio: 1:{(self.df['label'] == 0).sum() / (self.df['label'] == 1).sum():.1f}"
        )

        return self.df

    def split_data(
        self, train_split=0.8, val_split=0.1, test_split=0.1, random_seed=42
    ):
        """Split data into train/val/test with stratification"""
        if self.df is None:
            self.load_data()

        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            self.df,
            test_size=(val_split + test_split),
            stratify=self.df["label"],
            random_state=random_seed,
        )

        # Second split: val vs test
        val_size = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df["label"],
            random_state=random_seed,
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        print(f"\nData split:")
        print(
            f"Train: {len(self.train_df)} (RG: {(self.train_df['label'] == 1).sum()})"
        )
        print(f"Val: {len(self.val_df)} (RG: {(self.val_df['label'] == 1).sum()})")
        print(f"Test: {len(self.test_df)} (RG: {(self.test_df['label'] == 1).sum()})")

        return self.train_df, self.val_df, self.test_df

    def create_generators(self, batch_size=32, augment=True):
        """Create data generators for training"""
        if self.train_df is None:
            self.split_data()

        # Data augmentation for training
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                horizontal_flip=config.AUGMENTATION.get("horizontal_flip", True),
                vertical_flip=config.AUGMENTATION.get("vertical_flip", False),
                rotation_range=config.AUGMENTATION.get("rotation_range", 15),
                width_shift_range=config.AUGMENTATION.get("width_shift_range", 0.1),
                height_shift_range=config.AUGMENTATION.get("height_shift_range", 0.1),
                zoom_range=config.AUGMENTATION.get("zoom_range", 0.1),
                brightness_range=config.AUGMENTATION.get(
                    "brightness_range", [0.8, 1.2]
                ),
                fill_mode="constant",
                cval=0,
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # No augmentation for validation/test
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            self.train_df,
            x_col="image_path",
            y_col="label",
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=True,
            seed=config.RANDOM_SEED,
        )

        val_generator = val_datagen.flow_from_dataframe(
            self.val_df,
            x_col="image_path",
            y_col="label",
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=False,
        )

        test_generator = val_datagen.flow_from_dataframe(
            self.test_df,
            x_col="image_path",
            y_col="label",
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=False,
        )

        return train_generator, val_generator, test_generator


def preprocess_image(image_path, image_size=384):
    """Preprocess a single image"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = img / 255.0
    return img


def create_tf_dataset(df, batch_size=32, image_size=384, shuffle=False, augment=False):
    """Create TensorFlow dataset (alternative to ImageDataGenerator)"""
    image_paths = df["image_path"].values
    labels = df["label"].values

    def load_and_preprocess(path, label):
        img = preprocess_image(path, image_size)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=config.RANDOM_SEED)

    dataset = dataset.map(
        lambda path, label: tf.py_function(
            load_and_preprocess, [path, label], [tf.float32, tf.int64]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
