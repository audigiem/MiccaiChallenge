"""
Inference script for AIROGS glaucoma detection
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import config
from evaluation import compute_partial_auc, compute_sensitivity_at_specificity


def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_image(image_path, image_size=384):
    """Preprocess single image for inference"""
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Resize
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_single_image(model, image_path, image_size=384):
    """
    Predict glaucoma for a single image

    Returns:
        O1: score for likelihood of referable glaucoma (0-1)
        O2: binary glaucoma decision (0 or 1)
        O3: binary decision gradable/ungradable (always 1 for baseline)
        O4: ungradability score (placeholder for baseline)
    """
    # Preprocess
    img = preprocess_image(image_path, image_size)

    # Predict
    pred_proba = model.predict(img, verbose=0)[0][0]

    # Output 1: Glaucoma probability score
    O1 = float(pred_proba)

    # Output 2: Binary decision (using 0.5 threshold)
    O2 = int(pred_proba >= 0.5)

    # Output 3: Gradability (baseline assumes all images are gradable)
    O3 = 1

    # Output 4: Ungradability score (baseline uses prediction entropy as proxy)
    # Higher entropy = more uncertainty = potentially ungradable
    entropy = -pred_proba * np.log(pred_proba + 1e-7) - (1 - pred_proba) * np.log(
        1 - pred_proba + 1e-7
    )
    O4 = float(entropy)

    return O1, O2, O3, O4


def predict_batch(model, image_dir, output_csv, image_size=384):
    """
    Predict glaucoma for a batch of images

    Args:
        model: Trained model
        image_dir: Directory containing images
        output_csv: Path to save predictions
        image_size: Image size for preprocessing
    """
    # Get all image files
    image_files = [
        f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    image_files.sort()

    print(f"Found {len(image_files)} images in {image_dir}")

    results = []

    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(image_files)}...")

        image_path = os.path.join(image_dir, image_file)
        image_id = os.path.splitext(image_file)[0]

        try:
            O1, O2, O3, O4 = predict_single_image(model, image_path, image_size)

            results.append(
                {
                    "challenge_id": image_id,
                    "glaucoma_score": O1,
                    "glaucoma_binary": O2,
                    "gradable": O3,
                    "ungradability_score": O4,
                }
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results.append(
                {
                    "challenge_id": image_id,
                    "glaucoma_score": 0.0,
                    "glaucoma_binary": 0,
                    "gradable": 0,
                    "ungradability_score": 1.0,
                }
            )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to {output_csv}")

    # Print summary
    print(f"\nPrediction Summary:")
    print(f"Total images: {len(df)}")
    print(f"Predicted RG (glaucoma): {df['glaucoma_binary'].sum()}")
    print(f"Predicted NRG (no glaucoma): {(df['glaucoma_binary'] == 0).sum()}")
    print(f"Mean glaucoma score: {df['glaucoma_score'].mean():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="AIROGS inference script")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.h5 file)"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to single image for inference"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images for batch inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file for predictions",
    )
    parser.add_argument(
        "--image-size", type=int, default=384, help="Image size for preprocessing"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    if args.image:
        # Single image inference
        print(f"\nPredicting for single image: {args.image}")
        O1, O2, O3, O4 = predict_single_image(model, args.image, args.image_size)

        print("\nPrediction Results:")
        print(f"  O1 - Glaucoma score:       {O1:.4f}")
        print(f"  O2 - Glaucoma binary:      {'RG' if O2 == 1 else 'NRG'}")
        print(f"  O3 - Gradable:             {'Yes' if O3 == 1 else 'No'}")
        print(f"  O4 - Ungradability score:  {O4:.4f}")

    elif args.image_dir:
        # Batch inference
        print(f"\nPredicting for images in: {args.image_dir}")
        df = predict_batch(model, args.image_dir, args.output, args.image_size)

    else:
        print("Error: Please provide either --image or --image-dir")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
