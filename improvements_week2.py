"""
Week 2 Improvements: Data-Focused Enhancements
Following expectations.txt guidelines - no architecture changes
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc


# ============================================================================
# 1. OPTIC DISC DETECTION AND CROPPING
# ============================================================================


def detect_optic_disc_hough(image_path, output_size=512):
    """
    Detect optic disc using Hough Circle Transform and crop around it.
    This is a simple, fast preprocessing step used by many challenge winners.

    Args:
        image_path: Path to fundus image
        output_size: Size of output crop (default 512x512)

    Returns:
        Cropped image centered on optic disc (or center if detection fails)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detect circles (optic disc is typically bright and circular)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=h // 4,
        param1=50,
        param2=30,
        minRadius=h // 20,
        maxRadius=h // 5,
    )

    # Determine crop center
    if circles is not None:
        # Use brightest circle as optic disc
        circles = np.uint16(np.around(circles))
        center_x, center_y = circles[0][0][:2]
    else:
        # Fallback to image center
        center_x, center_y = w // 2, h // 2

    # Calculate crop coordinates
    half_size = output_size // 2
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(w, center_x + half_size)
    y2 = min(h, center_y + half_size)

    # Crop and resize
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (output_size, output_size))

    return crop


# ============================================================================
# 2. CLAHE PREPROCESSING FOR ILLUMINATION NORMALIZATION
# ============================================================================


def apply_clahe_preprocessing(image, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to normalize illumination across different acquisition devices.

    This is critical for cross-center robustness in the AIROGS challenge.

    Args:
        image: Input image (numpy array or tensor)
        clip_limit: CLAHE clip limit
        tile_size: Size of grid for histogram equalization

    Returns:
        Preprocessed image
    """
    # Convert to numpy if tensor
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result.astype(np.float32) / 255.0


# ============================================================================
# 3. FOCAL LOSS FOR EXTREME CLASS IMBALANCE
# ============================================================================


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    Reduces loss contribution from easy examples, focusing on hard cases.

    Original paper: Lin et al., "Focal Loss for Dense Object Detection"

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for class balance

    Returns:
        Loss function
    """

    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = K.pow(1 - pt, gamma)

        # Binary cross-entropy
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # Apply focal weighting
        loss = alpha * focal_weight * bce

        return K.mean(loss)

    return focal_loss_fixed


# ============================================================================
# 4. TEST-TIME AUGMENTATION (TTA)
# ============================================================================


def predict_with_tta(model, image, n_augmentations=5):
    """
    Perform test-time augmentation to improve prediction stability.

    Applies multiple augmentations and averages predictions.
    This is a common technique used by challenge winners.

    Args:
        model: Trained Keras model
        image: Input image (H, W, 3)
        n_augmentations: Number of augmented versions to average

    Returns:
        Average prediction
    """
    predictions = []

    # Original prediction
    pred = model.predict(np.expand_dims(image, 0), verbose=0)[0]
    predictions.append(pred)

    # Augmented predictions
    for _ in range(n_augmentations - 1):
        aug_image = augment_image(image)
        pred = model.predict(np.expand_dims(aug_image, 0), verbose=0)[0]
        predictions.append(pred)

    return np.mean(predictions, axis=0)


def augment_image(image):
    """
    Apply random augmentation for TTA.

    Args:
        image: Input image

    Returns:
        Augmented image
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)

    # Random vertical flip (fundus images can be rotated)
    if np.random.rand() > 0.5:
        image = np.flipud(image)

    # Random rotation (±15 degrees)
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(
        image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Random brightness adjustment
    brightness = np.random.uniform(0.9, 1.1)
    image = np.clip(image * brightness, 0, 1)

    return image


# ============================================================================
# 5. OPTIMIZED THRESHOLD SELECTION FOR PARTIAL AUC
# ============================================================================


def find_optimal_threshold_for_pauc(
    y_true, y_pred_proba, target_specificity=0.95, specificity_range=(0.90, 1.0)
):
    """
    Find optimal threshold that maximizes sensitivity at target specificity.
    This is crucial for the challenge's partial AUC metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target_specificity: Target specificity (default 0.95)
        specificity_range: Range for partial AUC (default 90-100%)

    Returns:
        optimal_threshold, sensitivity, actual_specificity
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    specificity = 1 - fpr

    # Find thresholds within target specificity range
    mask = (specificity >= specificity_range[0]) & (specificity <= specificity_range[1])

    if not np.any(mask):
        print("⚠️  Warning: No thresholds found in specificity range")
        # Fallback: find closest to target
        idx = np.argmin(np.abs(specificity - target_specificity))
        return thresholds[idx], tpr[idx], specificity[idx]

    # Among valid thresholds, find the one maximizing sensitivity
    valid_indices = np.where(mask)[0]

    # Find threshold closest to target specificity with highest sensitivity
    target_idx = None
    best_sensitivity = -1
    best_spec_diff = float("inf")

    for idx in valid_indices:
        spec_diff = abs(specificity[idx] - target_specificity)
        if spec_diff < best_spec_diff or (
            spec_diff == best_spec_diff and tpr[idx] > best_sensitivity
        ):
            best_spec_diff = spec_diff
            best_sensitivity = tpr[idx]
            target_idx = idx

    optimal_threshold = thresholds[target_idx]
    optimal_sensitivity = tpr[target_idx]
    actual_specificity = specificity[target_idx]

    return optimal_threshold, optimal_sensitivity, actual_specificity


# ============================================================================
# 6. ENSEMBLE PREDICTIONS
# ============================================================================


def ensemble_predict(models, image, use_tta=True):
    """
    Ensemble predictions from multiple models.

    Args:
        models: List of trained models
        image: Input image
        use_tta: Whether to use TTA for each model

    Returns:
        Average prediction across models
    """
    predictions = []

    for model in models:
        if use_tta:
            pred = predict_with_tta(model, image, n_augmentations=5)
        else:
            pred = model.predict(np.expand_dims(image, 0), verbose=0)[0]
        predictions.append(pred)

    return np.mean(predictions, axis=0)


# ============================================================================
# 7. EVALUATION UTILITIES
# ============================================================================


def calculate_partial_auc(y_true, y_pred_proba, specificity_range=(0.90, 1.0)):
    """
    Calculate partial AUC in the specificity range [90%, 100%].
    This is the primary challenge metric (α).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        specificity_range: Specificity range for partial AUC

    Returns:
        Partial AUC value (normalized to [0, 1])
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Convert to specificity
    specificity = 1 - fpr

    # Filter to specificity range
    mask = (specificity >= specificity_range[0]) & (specificity <= specificity_range[1])

    if not np.any(mask):
        return 0.0

    # Calculate partial AUC
    fpr_filtered = fpr[mask]
    tpr_filtered = tpr[mask]

    # Sort by fpr
    sort_idx = np.argsort(fpr_filtered)
    fpr_sorted = fpr_filtered[sort_idx]
    tpr_sorted = tpr_filtered[sort_idx]

    # Calculate AUC in FPR range
    fpr_min = 1 - specificity_range[1]
    fpr_max = 1 - specificity_range[0]

    partial_auc = auc(fpr_sorted, tpr_sorted)

    # Normalize to [0, 1]
    max_possible_auc = fpr_max - fpr_min
    normalized_pauc = partial_auc / max_possible_auc if max_possible_auc > 0 else 0.0

    return normalized_pauc


if __name__ == "__main__":
    print("Week 2 Improvements Module")
    print("=" * 60)
    print("\nAvailable improvements:")
    print("1. Optic disc detection and cropping")
    print("2. CLAHE preprocessing for illumination normalization")
    print("3. Focal loss for extreme class imbalance")
    print("4. Test-time augmentation (TTA)")
    print("5. Optimized threshold selection for partial AUC")
    print("6. Ensemble predictions")
    print("\nThese improvements focus on data handling, not architecture changes.")
    print("See expectations.txt - Week 2 requirements.")
