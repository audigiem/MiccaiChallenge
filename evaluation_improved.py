"""
Improved evaluation script with Week 2 enhancements
Uses TTA and optimized thresholding to improve pAUC and sensitivity
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from datetime import datetime
import json

# Import local modules
import config
from dataset import AIROGSDataset
from improvements_week2 import (
    predict_with_tta,
    find_optimal_threshold_for_pauc,
    calculate_partial_auc,
    apply_clahe_preprocessing
)


def evaluate_with_improvements(model_path, use_tta=True, use_clahe=False):
    """
    Evaluate model with Week 2 improvements.
    
    Args:
        model_path: Path to trained model
        use_tta: Whether to use test-time augmentation
        use_clahe: Whether to apply CLAHE preprocessing
    """
    print("\n" + "=" * 70)
    print("AIROGS EVALUATION - WITH WEEK 2 IMPROVEMENTS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Test-Time Augmentation: {'ENABLED' if use_tta else 'DISABLED'}")
    print(f"CLAHE Preprocessing: {'ENABLED' if use_clahe else 'DISABLED'}")
    print("=" * 70 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully\n")
    
    # Load test data
    print("📊 Loading test dataset...")
    dataset = AIROGSDataset(
        labels_csv=config.TRAIN_LABELS_CSV,
        images_dir=config.TRAIN_IMAGES_DIR
    )
    dataset.load_data()
    
    # Use all data as test (same as baseline evaluation)
    _, _, test_df = dataset.split_data(
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
        random_seed=config.RANDOM_SEED
    )
    
    print(f"Test set size: {len(test_df)} images\n")
    
    # Make predictions
    print("🔮 Making predictions...")
    y_true = test_df["label"].values
    y_pred_proba = []
    
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing image {idx + 1}/{len(test_df)}...")
        
        # Load image
        img = load_img(row["image_path"], target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        img = img_to_array(img) / 255.0
        
        # Apply CLAHE if enabled
        if use_clahe:
            img = apply_clahe_preprocessing(img)
        
        # Predict with or without TTA
        if use_tta:
            pred = predict_with_tta(model, img, n_augmentations=5)
        else:
            pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        
        y_pred_proba.append(pred[0])
    
    y_pred_proba = np.array(y_pred_proba)
    print("✅ Predictions complete\n")
    
    # Optimize threshold for partial AUC
    print("🎯 Optimizing threshold for pAUC...")
    optimal_threshold, sensitivity, actual_spec = find_optimal_threshold_for_pauc(
        y_true, y_pred_proba, target_specificity=0.95
    )
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Actual specificity: {actual_spec:.4f}\n")
    
    # Calculate metrics
    print("📈 Calculating metrics...\n")
    
    # Challenge metrics
    pauc = calculate_partial_auc(y_true, y_pred_proba, specificity_range=(0.90, 1.0))
    
    # Standard metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # Predictions with optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("=" * 70)
    print("IMPROVED EVALUATION RESULTS")
    print("=" * 70)
    print("\nCHALLENGE METRICS:")
    print(f"  α - Partial AUC (90-100% spec): {pauc:.4f}")
    print(f"  β - Sensitivity @ 95% spec:     {sensitivity:.4f}")
    print(f"      (actual specificity:         {actual_spec:.4f})")
    print(f"      (threshold used:             {optimal_threshold:.4f})")
    print("\nSTANDARD METRICS:")
    print(f"  AUC-ROC:      {auc_roc:.4f}")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1-Score:     {f1:.4f}")
    print(f"  Specificity:  {specificity:.4f}")
    print("\nCONFUSION MATRIX:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Negatives: {fn}")
    print("=" * 70 + "\n")
    
    # Compare with baseline
    print("📊 COMPARISON WITH BASELINE:")
    print(f"  Baseline pAUC:        0.6246")
    print(f"  Improved pAUC:        {pauc:.4f}  {'✅ +' + str(round((pauc - 0.6246) * 100, 2)) + '%' if pauc > 0.6246 else '❌'}")
    print(f"  Baseline Sensitivity: 0.6778")
    print(f"  Improved Sensitivity: {sensitivity:.4f}  {'✅ +' + str(round((sensitivity - 0.6778) * 100, 2)) + '%' if sensitivity > 0.6778 else '❌'}")
    print("=" * 70 + "\n")
    
    # Save results
    results = {
        "improvements": {
            "tta": use_tta,
            "clahe": use_clahe
        },
        "challenge_metrics": {
            "partial_auc": float(pauc),
            "sensitivity_at_95_spec": float(sensitivity),
            "actual_specificity": float(actual_spec),
            "optimal_threshold": float(optimal_threshold)
        },
        "standard_metrics": {
            "auc_roc": float(auc_roc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity)
        },
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }
    
    output_file = os.path.join(
        os.path.dirname(model_path),
        f"improved_evaluation_{'tta_' if use_tta else ''}{'clahe_' if use_clahe else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model with Week 2 improvements")
    parser.add_argument("model_path", type=str, help="Path to trained model (.keras)")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--clahe", action="store_true", help="Use CLAHE preprocessing")
    
    args = parser.parse_args()
    
    evaluate_with_improvements(args.model_path, use_tta=args.tta, use_clahe=args.clahe)
