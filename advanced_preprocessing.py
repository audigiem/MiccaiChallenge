"""
Advanced preprocessing utilities for glaucoma detection
Includes:
- Optic disk detection and cropping
- Green channel extraction (better for fundus)
- Advanced CLAHE
- Vessel enhancement
"""

import cv2
import numpy as np
from skimage import exposure, filters
from scipy import ndimage


def extract_green_channel(image):
    """
    Extract green channel from RGB fundus image
    Green channel has best contrast for blood vessels and optic disk
    """
    if len(image.shape) == 3:
        return image[:, :, 1]
    return image


def detect_optic_disk_simple(image, visualize=False):
    """
    Simple optic disk detection using brightness and circular Hough transform

    Args:
        image: RGB or grayscale fundus image
        visualize: If True, return annotated image

    Returns:
        (center_x, center_y, radius) or None if not found
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Find brightest region (optic disk is typically brightest)
    # Use top 1% brightness threshold
    threshold = np.percentile(blurred, 99)
    bright_mask = (blurred > threshold).astype(np.uint8) * 255

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit circle to contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    # Validate: optic disk should be 5-15% of image width
    image_width = image.shape[1]
    if radius < 0.05 * image_width or radius > 0.20 * image_width:
        # Try alternative: find moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            # Estimate radius from area
            area = cv2.contourArea(largest_contour)
            radius = np.sqrt(area / np.pi)

    if visualize:
        vis_img = image.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        cv2.circle(vis_img, (int(x), int(y)), int(radius), (0, 255, 0), 3)
        cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        return (int(x), int(y), int(radius)), vis_img

    return (int(x), int(y), int(radius))


def crop_around_optic_disk(image, od_info, crop_factor=2.5):
    """
    Crop image around optic disk region

    Args:
        image: Input image
        od_info: (center_x, center_y, radius) from detect_optic_disk_simple
        crop_factor: How much larger than optic disk to crop (default 2.5x)

    Returns:
        Cropped image centered on optic disk
    """
    if od_info is None:
        return image

    x, y, radius = od_info
    crop_size = int(radius * crop_factor * 2)

    # Ensure crop size is reasonable
    crop_size = min(crop_size, image.shape[0], image.shape[1])
    crop_size = max(crop_size, 100)  # Minimum 100px

    # Calculate crop coordinates
    x1 = max(0, int(x - crop_size // 2))
    y1 = max(0, int(y - crop_size // 2))
    x2 = min(image.shape[1], int(x + crop_size // 2))
    y2 = min(image.shape[0], int(y + crop_size // 2))

    # Crop
    cropped = image[y1:y2, x1:x2]

    return cropped


def enhance_vessels(image, sigma=1.5):
    """
    Enhance blood vessels using Frangi filter (vessel enhancement)

    Args:
        image: Grayscale image
        sigma: Scale parameter for vessel detection

    Returns:
        Vessel-enhanced image
    """
    # Frangi filter for vessel detection
    # Detects tubular structures (blood vessels)
    try:
        from skimage.filters import frangi

        enhanced = frangi(image, sigmas=range(1, 4), black_ridges=False)
        # Normalize to 0-255
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255
        return enhanced.astype(np.uint8)
    except:
        # Fallback: simple contrast enhancement
        return image


def apply_advanced_clahe(image, clip_limit=2.0, tile_size=8, color_mode="LAB"):
    """
    Advanced CLAHE with proper color space handling

    Args:
        image: RGB image (0-255 or 0-1)
        clip_limit: CLAHE clip limit
        tile_size: Grid size for CLAHE
        color_mode: 'LAB', 'HSV', or 'RGB'

    Returns:
        CLAHE-enhanced image in original format
    """
    # Ensure uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    if color_mode == "LAB":
        # Convert to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif color_mode == "HSV":
        # Convert to HSV, apply CLAHE to V channel
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    else:  # RGB
        # Apply to each channel
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = clahe.apply(image[:, :, i])

    return result


def preprocess_fundus_advanced(
    image_path,
    target_size=384,
    use_od_crop=True,
    use_clahe=True,
    use_vessel_enhance=False,
):
    """
    Complete advanced preprocessing pipeline

    Args:
        image_path: Path to fundus image
        target_size: Output size
        use_od_crop: Whether to detect and crop around optic disk
        use_clahe: Whether to apply CLAHE
        use_vessel_enhance: Whether to enhance vessels

    Returns:
        Preprocessed image ready for model
    """
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path

    # 1. Optic disk detection and cropping (optional)
    if use_od_crop:
        od_info = detect_optic_disk_simple(image)
        if od_info is not None:
            image = crop_around_optic_disk(image, od_info, crop_factor=3.0)

    # 2. Resize to target size
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 3. CLAHE enhancement
    if use_clahe:
        image = apply_advanced_clahe(
            image, clip_limit=2.0, tile_size=8, color_mode="LAB"
        )

    # 4. Vessel enhancement (optional, experimental)
    if use_vessel_enhance:
        green_channel = extract_green_channel(image)
        vessels = enhance_vessels(green_channel)
        # Blend with original
        image[:, :, 1] = (image[:, :, 1] * 0.7 + vessels * 0.3).astype(np.uint8)

    # 5. Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def batch_preprocess_dataset(image_paths, output_dir, **kwargs):
    """
    Preprocess entire dataset and save to disk

    Args:
        image_paths: List of image paths
        output_dir: Where to save preprocessed images
        **kwargs: Arguments for preprocess_fundus_advanced
    """
    import os
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Preprocessing images"):
        try:
            # Preprocess
            img = preprocess_fundus_advanced(img_path, **kwargs)

            # Save
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)

            # Convert back to uint8 for saving
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Preprocessing complete! Saved to {output_dir}")


if __name__ == "__main__":
    # Test preprocessing
    import sys

    if len(sys.argv) > 1:
        test_image = sys.argv[1]

        # Load image
        img = cv2.imread(test_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print("Testing optic disk detection...")
        od_info, vis_img = detect_optic_disk_simple(img, visualize=True)

        if od_info:
            print(f"✅ Optic disk found at: {od_info}")
            cv2.imwrite(
                "optic_disk_detection.jpg", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            )
            print("Saved visualization to optic_disk_detection.jpg")
        else:
            print("❌ Optic disk not detected")

        print("\nTesting full preprocessing pipeline...")
        processed = preprocess_fundus_advanced(
            img,
            target_size=384,
            use_od_crop=True,
            use_clahe=True,
            use_vessel_enhance=False,
        )

        # Save processed image
        processed_uint8 = (processed * 255).astype(np.uint8)
        cv2.imwrite(
            "preprocessed_advanced.jpg",
            cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR),
        )
        print("Saved preprocessed image to preprocessed_advanced.jpg")

        print("\n✅ Preprocessing test complete!")
    else:
        print("Usage: python advanced_preprocessing.py <path_to_fundus_image>")
