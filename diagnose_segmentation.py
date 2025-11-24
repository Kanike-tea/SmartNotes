#!/usr/bin/env python3
"""
Diagnose line segmentation issues
Shows detailed info about what's happening in pipeline
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import numpy as np
from preprocessing.line_segment import segment_lines


def diagnose_image(image_path):
    """
    Diagnose what happens during line segmentation
    """
    print(f"\n{'='*70}")
    print(f"LINE SEGMENTATION DIAGNOSIS")
    print(f"{'='*70}\n")
    
    print(f"Image: {Path(image_path).name}\n")
    
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ FAILED: Could not load image")
        return
    
    h, w = img.shape
    print(f"✓ Image loaded: {w}×{h} pixels")
    print(f"  Size: {w*h:,} total pixels")
    
    # Check image quality
    mean_val = np.mean(img)
    std_val = np.std(img)
    print(f"  Intensity: mean={mean_val:.1f}, std={std_val:.1f}")
    print(f"  Min/Max: {np.min(img)}/{np.max(img)}")
    
    # Downsample if needed
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_orig = img.copy()
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"\n⚠️  DOWNSAMPLED: {w}×{h} → {new_w}×{new_h} (scale={scale:.2f})")
        h, w = img.shape
    
    # Try segmentation
    print(f"\n{'─'*70}")
    print(f"RUNNING SEGMENTATION...")
    print(f"{'─'*70}\n")
    
    try:
        lines = segment_lines(str(image_path), debug=True)
        print(f"✓ Segmentation completed")
        print(f"  Found: {len(lines)} lines")
        
        if len(lines) == 0:
            print(f"\n❌ PROBLEM: No lines found!")
            print(f"  Possible causes:")
            print(f"    1. Image is all white/blank")
            print(f"    2. Text too small or low contrast")
            print(f"    3. Threshold parameters wrong for this document")
            print(f"    4. No horizontal text regions detected")
            
            # Additional diagnostics
            print(f"\n  Diagnostics:")
            
            # Check if image is mostly white/blank
            white_pixels = np.sum(img > 200)
            white_percent = 100 * white_pixels / (w * h)
            print(f"    - White pixels: {white_percent:.1f}%")
            
            # Check contrast
            if std_val < 10:
                print(f"    - ⚠️  Very low contrast (std={std_val:.1f})")
            
            # Try with different threshold
            print(f"\n  Trying alternative thresholds...")
            for thresh_val in [50, 100, 150, 200]:
                _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                h_contours = [c for c in contours if cv2.contourArea(c) > 50]
                print(f"    - Threshold {thresh_val}: {len(h_contours)} potential lines")
                
        else:
            print(f"\n  Line details:")
            for i, line in enumerate(lines[:3], 1):
                h_l, w_l = line.shape
                print(f"    Line {i}: {w_l}×{h_l} pixels")
            if len(lines) > 3:
                print(f"    ... and {len(lines)-3} more lines")
    
    except Exception as e:
        print(f"❌ SEGMENTATION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find a test image
        test_images = list(Path("datasets").rglob("*.png"))[:1]
        if not test_images:
            test_images = list(Path("datasets").rglob("*.jpg"))[:1]
        
        if not test_images:
            print("Usage: python diagnose_segmentation.py <image_path>")
            print("No test images found in datasets/")
            sys.exit(1)
        
        image_path = test_images[0]
    else:
        image_path = sys.argv[1]
    
    diagnose_image(str(image_path))
