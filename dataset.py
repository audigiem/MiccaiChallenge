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

        if isinstance(self.images_dir, list):
            dfs = []
            for img_dir in self.images_dir:
                # Extract dataset number from path (e.g., "dataset/0" -> "0")
                dataset_num = os.path.basename(img_dir)

                # Try to use specific labels file first (e.g., train_labels_0.csv)
                specific_labels = os.path.join(
                    os.path.dirname(self.labels_csv), f"train_labels_{dataset_num}.csv"
                )

                if os.path.exists(specific_labels):
                    labels_file = specific_labels
                    print(
                        f"\n   📋 Using specific labels: {os.path.basename(specific_labels)}"
                    )
                else:
                    labels_file = self.labels_csv
                    print(
                        f"\n   📋 Using global labels: {os.path.basename(labels_file)} (filtering for {img_dir})"
                    )

                df = pd.read_csv(labels_file)
                df["label"] = (df["class"] == "RG").astype(int)
                df["image_path"] = df["challenge_id"].apply(
                    lambda x: os.path.join(img_dir, f"{x}.jpg")
                )
                # Filter only existing images
                df = df[df["image_path"].apply(os.path.exists)]

                if len(df) > 0:
                    print(f"   Dataset: {img_dir}")
                    print(f"      Images: {len(df)}")
                    print(f"      RG: {(df['label'] == 1).sum()}")
                    print(f"      NRG: {(df['label'] == 0).sum()}")
                    dfs.append(df)
                else:
                    print(f"   ⚠️  Warning: No images found in {img_dir}")

            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.read_csv(self.labels_csv)

            # Convert labels to binary: RG=1, NRG=0
            self.df["label"] = (self.df["class"] == "RG").astype(int)

            # Add full image paths
            self.df["image_path"] = self.df["challenge_id"].apply(
                lambda x: os.path.join(self.images_dir, f"{x}.jpg")
            )

        # Always filter for existing images
        initial_count = len(self.df)
        self.df = self.df[self.df["image_path"].apply(os.path.exists)]
        filtered_count = initial_count - len(self.df)

        if filtered_count > 0:
            print(f"\n⚠️  Filtered out {filtered_count} images that don't exist on disk")

        print(f"\n✅ Total images loaded: {len(self.df)}")
        print(f"   RG (Glaucoma): {(self.df['label'] == 1).sum()}")
        print(f"   NRG (No Glaucoma): {(self.df['label'] == 0).sum()}")
        print(
            f"   Class imbalance ratio: 1:{(self.df['label'] == 0).sum() / (self.df['label'] == 1).sum():.1f}"
        )

        return self.df

    def split_data(
        self, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=None
    ):
        """
        Split labels dataframe into train/val/test and store them on the dataset instance.
        Handles the special case where train=0, val=0, test=1.0 by returning
        empty train/val DataFrames and the full dataframe as test.
        """
        total = train_split + val_split + test_split
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"train_split + val_split + test_split must sum to 1.0 (got {total})"
            )

        if self.df is None:
            raise AttributeError(
                "No dataframe found on dataset. Please call load_data() first."
            )
        df = self.df

        # Special-case: all data as test
        if (
            np.isclose(train_split, 0.0)
            and np.isclose(val_split, 0.0)
            and np.isclose(test_split, 1.0)
        ):
            empty = pd.DataFrame(columns=df.columns)
            self.train_df = empty
            self.val_df = empty
            self.test_df = df.copy().reset_index(drop=True)
            return self.train_df, self.val_df, self.test_df

        stratify_col = df["label"] if "label" in df.columns else None

        # First split: train vs (val+test)
        if np.isclose(train_split, 0.0):
            train_df = pd.DataFrame(columns=df.columns)
            temp_df = df.copy()
        else:
            temp_size = 1.0 - train_split
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_size,
                random_state=random_seed,
                stratify=stratify_col,
            )

        # If val_split is zero, remaining is test
        if np.isclose(val_split, 0.0):
            val_df = pd.DataFrame(columns=df.columns)
            test_df = temp_df
        else:
            # compute relative val size w.r.t. temp_df
            val_relative = val_split / (val_split + test_split)
            strat_temp = temp_df["label"] if "label" in temp_df.columns else None
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1.0 - val_relative),
                random_state=random_seed,
                stratify=strat_temp,
            )

        # Store on instance and return
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        return self.train_df, self.val_df, self.test_df

    def create_generators(self, batch_size=32, augment=True):
        """Create data generators for training"""
        # Ensure splits exist
        if self.train_df is None:
            self.split_data()

        def ensure_image_path_col(df):
            if df is None:
                return df
            # If column already present, return as-is
            if "image_path" in df.columns:
                return df
            # Try to build image_path from challenge_id
            if "challenge_id" in df.columns:
                df = df.copy()
                df["image_path"] = df["challenge_id"].apply(
                    lambda x: os.path.join(self.images_dir, f"{x}.jpg")
                )
                return df
            # If neither column is present, return original (flow_from_dataframe will raise a clear error later)
            return df

        train_df = ensure_image_path_col(self.train_df)
        val_df = ensure_image_path_col(self.val_df)
        test_df = ensure_image_path_col(self.test_df)

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

        def make_generator(datagen, df, shuffle):
            if df is None or len(df) == 0:
                return None
            # Validate required columns before calling keras
            if "image_path" not in df.columns or "label" not in df.columns:
                raise KeyError(
                    "DataFrame must contain 'image_path' and 'label' columns before creating generator."
                )
            return datagen.flow_from_dataframe(
                df,
                x_col="image_path",
                y_col="label",
                target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                batch_size=batch_size,
                class_mode="raw",
                shuffle=shuffle,
                seed=config.RANDOM_SEED,
            )

        train_generator = make_generator(train_datagen, train_df, shuffle=True)
        val_generator = make_generator(val_datagen, val_df, shuffle=False)
        test_generator = make_generator(val_datagen, test_df, shuffle=False)

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
