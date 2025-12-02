"""
Utility functions for AIROGS challenge
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf


def visualize_samples(dataset_df, num_samples=9, save_path=None):
    """
    Visualize random samples from dataset with labels

    Args:
        dataset_df: DataFrame with image_path and label columns
        num_samples: Number of samples to display
        save_path: Path to save figure (optional)
    """
    samples = dataset_df.sample(n=num_samples)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= num_samples:
            break

        img = Image.open(row["image_path"])
        axes[idx].imshow(img)

        label = "RG (Glaucoma)" if row["label"] == 1 else "NRG (No Glaucoma)"
        axes[idx].set_title(f"{row['challenge_id']}\n{label}")
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Samples visualization saved to {save_path}")

    return fig


def plot_training_history(history_dict, save_path=None):
    """
    Plot training history

    Args:
        history_dict: Dictionary containing training history
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history_dict["loss"], label="Train Loss")
    axes[0, 0].plot(history_dict["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history_dict["accuracy"], label="Train Accuracy")
    axes[0, 1].plot(history_dict["val_accuracy"], label="Val Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC
    axes[1, 0].plot(history_dict["auc"], label="Train AUC")
    axes[1, 0].plot(history_dict["val_auc"], label="Val AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_title("Training and Validation AUC")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 1].plot(history_dict["precision"], label="Train Precision", linestyle="--")
    axes[1, 1].plot(
        history_dict["val_precision"], label="Val Precision", linestyle="--"
    )
    axes[1, 1].plot(history_dict["recall"], label="Train Recall", linestyle="-.")
    axes[1, 1].plot(history_dict["val_recall"], label="Val Recall", linestyle="-.")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Precision and Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    return fig


def visualize_predictions(
    model, dataset_df, num_samples=9, image_size=384, save_path=None
):
    """
    Visualize model predictions on random samples

    Args:
        model: Trained Keras model
        dataset_df: DataFrame with image_path and label columns
        num_samples: Number of samples to display
        image_size: Image size for preprocessing
        save_path: Path to save figure (optional)
    """
    samples = dataset_df.sample(n=num_samples)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= num_samples:
            break

        # Load and preprocess image
        img = Image.open(row["image_path"]).convert("RGB")
        img_resized = img.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred_proba = model.predict(img_array, verbose=0)[0][0]
        pred_label = "RG" if pred_proba >= 0.5 else "NRG"
        true_label = "RG" if row["label"] == 1 else "NRG"

        # Display
        axes[idx].imshow(img_resized)

        # Color code: green if correct, red if incorrect
        color = "green" if pred_label == true_label else "red"

        axes[idx].set_title(
            f"True: {true_label} | Pred: {pred_label}\n"
            f"Confidence: {pred_proba:.3f}",
            color=color,
            fontweight="bold",
        )
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Predictions visualization saved to {save_path}")

    return fig


def check_data_integrity(images_dir, labels_csv):
    """
    Check data integrity: missing files, corrupt images, etc.

    Args:
        images_dir: Directory containing images
        labels_csv: Path to labels CSV file
    """
    import pandas as pd

    print("\n🔍 Checking data integrity...")

    # Load labels
    df = pd.read_csv(labels_csv)
    print(f"Total entries in CSV: {len(df)}")

    # Check for missing image files
    missing_files = []
    corrupt_files = []

    for idx, row in df.iterrows():
        image_path = os.path.join(images_dir, f"{row['challenge_id']}.jpg")

        if not os.path.exists(image_path):
            missing_files.append(row["challenge_id"])
        else:
            # Try to open image to check if corrupt
            try:
                img = Image.open(image_path)
                img.verify()
            except Exception as e:
                corrupt_files.append((row["challenge_id"], str(e)))

        if (idx + 1) % 10000 == 0:
            print(f"Checked {idx + 1}/{len(df)} files...")

    print("\n✅ Data Integrity Check Results:")
    print(f"Total files checked: {len(df)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupt files: {len(corrupt_files)}")

    if missing_files:
        print(f"\nFirst 10 missing files: {missing_files[:10]}")

    if corrupt_files:
        print(f"\nFirst 10 corrupt files: {corrupt_files[:10]}")

    if not missing_files and not corrupt_files:
        print("✅ All files are present and valid!")

    return missing_files, corrupt_files


def analyze_class_distribution(labels_csv):
    """
    Analyze and visualize class distribution

    Args:
        labels_csv: Path to labels CSV file
    """
    import pandas as pd

    df = pd.read_csv(labels_csv)

    # Count classes
    class_counts = df["class"].value_counts()

    print("\n📊 Class Distribution:")
    print(class_counts)
    print(
        f"\nImbalance ratio (NRG:RG): {class_counts['NRG'] / class_counts['RG']:.2f}:1"
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    class_counts.plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Pie chart
    axes[1].pie(
        class_counts.values,
        labels=class_counts.index,
        autopct="%1.1f%%",
        colors=["#3498db", "#e74c3c"],
        startangle=90,
    )
    axes[1].set_title("Class Distribution (%)", fontsize=14, fontweight="bold")

    plt.tight_layout()

    return fig


def create_submission_format(predictions_df, output_path):
    """
    Create submission file in AIROGS challenge format

    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save submission file
    """
    # AIROGS format:
    # challenge_id, O1 (glaucoma_score), O2 (binary), O3 (gradable), O4 (ungradability)

    submission = predictions_df[
        [
            "challenge_id",
            "glaucoma_score",
            "glaucoma_binary",
            "gradable",
            "ungradability_score",
        ]
    ].copy()

    submission.columns = ["challenge_id", "O1", "O2", "O3", "O4"]

    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

    return submission
