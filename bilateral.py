#!/usr/bin/env python3
# bilateral_wrinkle_smoothing.py

import cv2
import numpy as np
import os
import argparse
import time
from pathlib import Path


def apply_bilateral_filter(image, diameter=75, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to smooth wrinkles while preserving edges."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def process_garment_regions(image, masks, verbose=False):
    """
    Apply bilateral filter to each garment region individually.

    Args:
        image: Full-color input image (BGR)
        masks: List of binary masks corresponding to each garment
        verbose: Whether to print timing and display intermediate outputs
    Returns:
        Filtered image with garment regions smoothed
    """
    output = image.copy()
    for idx, mask in enumerate(masks):
        garment = cv2.bitwise_and(image, image, mask=mask)
        smoothed = apply_bilateral_filter(garment)
        inv_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(output, output, mask=inv_mask)
        output = cv2.add(background, smoothed)
        if verbose:
            print(f"Processed garment {idx + 1}")
    return output


def load_masks(mask_dir, shape):
    """
    Load binary masks from a directory and resize to match image shape.

    Args:
        mask_dir: Directory containing mask images
        shape: Target (height, width) to resize masks
    Returns:
        List of binary masks
    """
    masks = []
    for file in sorted(Path(mask_dir).glob("*.png")):
        mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (shape[1], shape[0]))
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(binary)
    return masks


def visualize_comparison(original, filtered, out_path=None):
    """
    Show side-by-side comparison of original and filtered image.
    Optionally saves the comparison to a file.
    """
    concat = np.hstack((original, filtered))
    if out_path:
        cv2.imwrite(out_path, concat)
    cv2.imshow("Original (Left) vs Filtered (Right)", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Bilateral Filtering for Garment Wrinkle Smoothing")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (BGR)")
    parser.add_argument("--mask_dir", type=str, help="Directory containing binary garment masks")
    parser.add_argument("--out", type=str, required=True, help="Path to save filtered image")
    parser.add_argument("--compare", action="store_true", help="Show A/B comparison visualization")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.image}")

    start_time = time.time()

    if args.mask_dir:
        masks = load_masks(args.mask_dir, image.shape[:2])
        filtered = process_garment_regions(image, masks, verbose=args.verbose)
    else:
        # Apply to full image if no masks are provided
        filtered = apply_bilateral_filter(image)

    elapsed = (time.time() - start_time) * 1000
    if args.verbose:
        print(f"Filtering completed in {elapsed:.1f} ms")

    cv2.imwrite(args.out, filtered)

    if args.compare:
        visualize_comparison(image, filtered)


if __name__ == "__main__":
    main()
