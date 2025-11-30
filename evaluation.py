"""
Evaluation metrics for AIROGS challenge
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, confusion_matrix,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import config


def compute_partial_auc(y_true, y_pred, specificity_range=(0.9, 1.0)):
    """
    Compute partial AUC in specified specificity range

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        specificity_range: Tuple of (min_spec, max_spec)

    Returns:
        Partial AUC value
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Convert to specificity (specificity = 1 - fpr)
    specificity = 1 - fpr

    # Find indices within specificity range
    min_spec, max_spec = specificity_range

    # Reverse arrays because specificity decreases as fpr increases
    specificity_rev = specificity[::-1]
    tpr_rev = tpr[::-1]

    # Find indices in range
    mask = (specificity_rev >= min_spec) & (specificity_rev <= max_spec)

    if mask.sum() < 2:
        return 0.0

    # Extract relevant portion
    spec_range = specificity_rev[mask]
    tpr_range = tpr_rev[mask]

    # Sort by specificity for proper integration
    sorted_indices = np.argsort(spec_range)
    spec_range = spec_range[sorted_indices]
    tpr_range = tpr_range[sorted_indices]

    # Compute partial AUC using trapezoidal rule
    partial_auc = auc(spec_range, tpr_range)

    # Normalize by range width
    range_width = max_spec - min_spec
    normalized_pauc = partial_auc / range_width if range_width > 0 else 0.0

    return normalized_pauc


def compute_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95):
    """
    Compute sensitivity at a target specificity level

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        target_specificity: Target specificity (default 0.95)

    Returns:
        Sensitivity at target specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specificity = 1 - fpr

    # Find the threshold closest to target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))

    sensitivity = tpr[idx]
    actual_specificity = specificity[idx]
    threshold = thresholds[idx]

    return sensitivity, actual_specificity, threshold


def evaluate_model(model, test_generator, test_df):
    """
    Comprehensive model evaluation

    Args:
        model: Trained Keras model
        test_generator: Test data generator
        test_df: Test dataframe with true labels

    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(test_generator, verbose=1).flatten()
    y_true = test_df['label'].values

    # Standard metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # Partial AUC (90-100% specificity)
    pauc_score = compute_partial_auc(
        y_true, y_pred_proba,
        specificity_range=config.PAUC_RANGE
    )

    # Sensitivity at 95% specificity
    sensitivity_95, actual_spec_95, threshold_95 = compute_sensitivity_at_specificity(
        y_true, y_pred_proba,
        target_specificity=config.SPECIFICITY_THRESHOLD
    )

    # Binary predictions at threshold
    y_pred_binary = (y_pred_proba >= threshold_95).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results = {
        'auc': auc_score,
        'partial_auc': pauc_score,
        'sensitivity_at_95_specificity': sensitivity_95,
        'actual_specificity': actual_spec_95,
        'threshold_at_95_spec': threshold_95,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'y_true': y_true,
        'y_pred_proba': y_pred_proba,
        'y_pred_binary': y_pred_binary
    }

    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("AIROGS CHALLENGE - EVALUATION RESULTS")
    print("="*60)

    print("\n📊 CHALLENGE METRICS:")
    print(f"  α - Partial AUC (90-100% spec): {results['partial_auc']:.4f}")
    print(f"  β - Sensitivity @ 95% spec:     {results['sensitivity_at_95_specificity']:.4f}")
    print(f"      (actual specificity:         {results['actual_specificity']:.4f})")
    print(f"      (threshold used:             {results['threshold_at_95_spec']:.4f})")

    print("\n📈 STANDARD METRICS:")
    print(f"  AUC-ROC:      {results['auc']:.4f}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    print(f"  F1-Score:     {results['f1_score']:.4f}")
    print(f"  Specificity:  {results['specificity']:.4f}")

    print("\n🎯 CONFUSION MATRIX:")
    print(f"  True Positives:  {results['true_positives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  True Negatives:  {results['true_negatives']}")
    print(f"  False Negatives: {results['false_negatives']}")

    print("="*60 + "\n")


def plot_roc_curve(results, save_path=None):
    """Plot ROC curve with partial AUC highlighted"""
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
    specificity = 1 - fpr

    plt.figure(figsize=(10, 8))

    # Plot full ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {results["auc"]:.4f})')

    # Highlight partial AUC region (90-100% specificity = 0-10% FPR)
    mask = (fpr <= 0.1)
    plt.fill_between(fpr[mask], 0, tpr[mask], alpha=0.3, color='green',
                     label=f'Partial AUC (pAUC = {results["partial_auc"]:.4f})')

    # Plot sensitivity at 95% specificity point
    sens_95 = results['sensitivity_at_95_specificity']
    fpr_95 = 1 - results['actual_specificity']
    plt.plot(fpr_95, sens_95, 'ro', markersize=10,
             label=f'Sensitivity @ 95% Spec = {sens_95:.4f}')

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - AIROGS Glaucoma Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_confusion_matrix(results, save_path=None):
    """Plot confusion matrix"""
    cm = results['confusion_matrix']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['NRG', 'RG'],
                yticklabels=['NRG', 'RG'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - AIROGS Glaucoma Detection', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_prediction_distribution(results, save_path=None):
    """Plot distribution of predicted probabilities"""
    y_true = results['y_true']
    y_pred_proba = results['y_pred_proba']

    plt.figure(figsize=(10, 6))

    # Plot histograms for each class
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6, label='NRG (No Glaucoma)', color='blue')
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6, label='RG (Glaucoma)', color='red')

    # Add threshold line
    threshold = results['threshold_at_95_spec']
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2,
                label=f'Threshold @ 95% Spec = {threshold:.4f}')

    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution saved to {save_path}")

    plt.tight_layout()
    return plt.gcf()

