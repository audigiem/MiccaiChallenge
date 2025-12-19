"""
V4 Evaluation Script - ADVANCED PREPROCESSING
Evaluates V4 models with same preprocessing used during training
"""

import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
import json
import argparse

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import local modules
import config_v4 as config
from dataset_v4 import create_v4_generators
from dataset import AIROGSDataset
from evaluation_improved import compute_challenge_metrics, format_metrics_report


def evaluate_v4_model(model_path, use_tta=False, save_predictions=True):
    """
    Evaluate V4 model with advanced preprocessing

    Args:
        model_path: Path to trained V4 model
        use_tta: Whether to use test-time augmentation
        save_predictions: Whether to save predictions to CSV

    Returns:
        Dictionary with metrics and predictions
    """

    print("\n" + "=" * 70)
    print("AIROGS V4 EVALUATION - ADVANCED PREPROCESSING")
    print("=" * 70)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"TTA: {'ENABLED' if use_tta else 'DISABLED'}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print(f"\n📥 Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✅ Model loaded successfully")

    # Load evaluation dataset (dataset 5)
    print(f"\n📂 Loading evaluation dataset...")
    print(f"   Directory: {config.EVAL_IMAGES_DIR}")
    print(f"   Labels: {config.EVAL_LABELS_CSV}")

    eval_dataset = AIROGSDataset(
        labels_csv=config.EVAL_LABELS_CSV, images_dir=config.EVAL_IMAGES_DIR
    )
    eval_dataset.load_data()

    print(f"\n📊 Evaluation dataset:")
    print(f"   Total samples: {len(eval_dataset.df)}")

    label_counts = eval_dataset.df["label"].value_counts()
    print(f"   NRG (0): {label_counts.get(0, 0)}")
    print(f"   RG (1):  {label_counts.get(1, 0)}")

    # Create evaluation generator with V4 preprocessing
    print(f"\n🔄 Creating V4 evaluation generator...")
    print(f"   Advanced preprocessing:")
    print(f"      - Optic disk detection: {config.USE_OD_DETECTION}")
    print(f"      - Advanced CLAHE ({config.CLAHE_COLOR_SPACE}): {config.USE_CLAHE}")
    print(f"      - Vessel enhancement: {config.USE_VESSEL_ENHANCEMENT}")

    from dataset_v4 import V4AdvancedImageDataGenerator

    eval_datagen = V4AdvancedImageDataGenerator(
        use_od_crop=config.USE_OD_DETECTION,
        od_crop_factor=config.OD_CROP_FACTOR,
        use_clahe=config.USE_CLAHE,
        clahe_color_space=config.CLAHE_COLOR_SPACE,
        use_vessel_enhancement=config.USE_VESSEL_ENHANCEMENT,
        rescale=1.0 / 255.0,
    )

    eval_gen = eval_datagen.flow_from_dataframe(
        dataframe=eval_dataset.df,
        x_col="image_path",
        y_col="label",
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    # Get predictions
    print(f"\n🔮 Generating predictions...")

    if use_tta:
        print(f"   Using Test-Time Augmentation (4 augmentations)")

        # TTA with rotations
        tta_preds = []
        tta_configs = [
            {},  # Original
            {"horizontal_flip": True},
            {"rotation_range": 10},
            {"horizontal_flip": True, "rotation_range": 10},
        ]

        for i, tta_config in enumerate(tta_configs, 1):
            print(f"      Augmentation {i}/{len(tta_configs)}...")

            tta_datagen = V4AdvancedImageDataGenerator(
                use_od_crop=config.USE_OD_DETECTION,
                od_crop_factor=config.OD_CROP_FACTOR,
                use_clahe=config.USE_CLAHE,
                clahe_color_space=config.CLAHE_COLOR_SPACE,
                use_vessel_enhancement=config.USE_VESSEL_ENHANCEMENT,
                rescale=1.0 / 255.0,
                **tta_config,
            )

            tta_gen = tta_datagen.flow_from_dataframe(
                dataframe=eval_dataset.df,
                x_col="image_path",
                y_col="label",
                target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                batch_size=config.BATCH_SIZE,
                class_mode="binary",
                shuffle=False,
            )

            preds = model.predict(tta_gen, verbose=0)
            tta_preds.append(preds)

        # Average TTA predictions
        predictions = np.mean(tta_preds, axis=0).flatten()
        print(f"   ✅ TTA predictions averaged")
    else:
        predictions = model.predict(eval_gen, verbose=1).flatten()

    # Get true labels
    y_true = eval_dataset.df["label"].values

    # Print preprocessing statistics
    print("\n" + "=" * 70)
    eval_datagen.print_stats()
    print("=" * 70)

    # Compute metrics
    print(f"\n📊 Computing challenge metrics...")
    metrics = compute_challenge_metrics(
        y_true=y_true, y_pred=predictions, threshold=0.5
    )

    # Format report
    report = format_metrics_report(
        metrics, model_name=os.path.basename(model_path), tta_enabled=use_tta
    )

    print(report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("evaluation_results_v4", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model": os.path.basename(model_path),
                "tta": use_tta,
                "preprocessing": {
                    "optic_disk_detection": config.USE_OD_DETECTION,
                    "clahe_color_space": config.CLAHE_COLOR_SPACE,
                    "vessel_enhancement": config.USE_VESSEL_ENHANCEMENT,
                },
                "metrics": {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()
                },
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
    print(f"\n💾 Metrics saved: {metrics_path}")

    # Save predictions
    if save_predictions:
        predictions_df = eval_dataset.df.copy()
        predictions_df["prediction"] = predictions
        predictions_df["predicted_label"] = (predictions > metrics["threshold"]).astype(
            int
        )

        predictions_path = os.path.join(results_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"💾 Predictions saved: {predictions_path}")

    # Save text report
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"💾 Report saved: {report_path}")

    print(f"\n✅ V4 Evaluation completed: {results_dir}")

    return {"metrics": metrics, "predictions": predictions, "results_dir": results_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate V4 model with advanced preprocessing"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to trained V4 model (.keras)"
    )
    parser.add_argument(
        "--tta", action="store_true", help="Enable test-time augmentation"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save predictions CSV"
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        sys.exit(1)

    try:
        results = evaluate_v4_model(
            model_path=args.model_path,
            use_tta=args.tta,
            save_predictions=not args.no_save,
        )

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"pAUC: {results['metrics']['partial_auc']:.4f}")
        print(
            f"Sensitivity@95% Specificity: {results['metrics']['sensitivity_at_95_specificity']:.4f}"
        )
        print(f"AUC: {results['metrics']['auc']:.4f}")
        print(f"Threshold: {results['metrics']['threshold']:.4f}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ V4 Evaluation failed:")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
