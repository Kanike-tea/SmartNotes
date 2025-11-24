#!/usr/bin/env python3
"""
SmartNotes Image Diagnostics

Analyzes why specific image fails and provides:
1. Image statistics (mean, std, brightness, contrast)
2. Line segmentation analysis with debug images
3. Line-by-line OCR testing
4. Full page recognition
5. Actionable recommendations
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports

setup_imports()

import cv2
import numpy as np
import argparse


def diagnose_image(image_path, debug=False):
    """Run comprehensive image diagnostics"""
    
    print("\n" + "=" * 70)
    print("SmartNotes Image Diagnostics")
    print("=" * 70)
    
    # ==========================================
    # 1. LOAD AND VALIDATE IMAGE
    # ==========================================
    print(f"\nImage: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"✗ Error: Could not load image")
        return False
    
    print("✓ Image loaded successfully")
    h, w = img.shape
    print(f"  Dimensions: {w}x{h}")
    
    # ==========================================
    # 2. IMAGE STATISTICS
    # ==========================================
    print("\nImage Statistics:")
    
    mean_val = np.mean(img)
    std_val = np.std(img)
    min_val = np.min(img)
    max_val = np.max(img)
    
    print(f"  Mean intensity: {mean_val:.2f}")
    print(f"  Std deviation: {std_val:.2f}")
    print(f"  Min/Max: {min_val}/{max_val}")
    
    # Quality assessment
    if std_val < 20:
        print("  ⚠ Very low contrast - may be difficult to segment")
    elif std_val > 80:
        print("  ⚠ Very high contrast - may be over-saturated")
    else:
        print("  ✓ Good contrast")
    
    if mean_val < 50:
        print("  ⚠ Image very dark - may need enhancement")
    elif mean_val > 200:
        print("  ✓ Good brightness")
    else:
        print("  ⚠ Image somewhat bright")
    
    # ==========================================
    # 3. LINE SEGMENTATION
    # ==========================================
    print("\nLine Segmentation Test")
    print("=" * 70)
    
    from preprocessing.line_segment import segment_lines
    
    lines = segment_lines(image_path, debug=True)
    
    if len(lines) == 0:
        print("✗ No lines detected")
        print("\nRecommendations:")
        print("1. Check image quality - may be too dark/bright")
        print("2. Verify image contains actual text")
        print("3. Try preprocessing the image manually")
        return False
    
    print(f"\n✓ Detected {len(lines)} lines")
    
    # ==========================================
    # 4. SAVE DEBUG IMAGES
    # ==========================================
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    print(f"\n✓ Saved first 10 lines to {debug_dir}/")
    for i, line in enumerate(lines[:10]):
        cv2.imwrite(str(debug_dir / f"line_{i:03d}.png"), line)
    
    # ==========================================
    # 5. LINE-BY-LINE OCR TEST
    # ==========================================
    print("\nLine-by-Line OCR Test")
    print("=" * 70)
    
    from preprocessing.recognize import OCRRecognizer
    
    recognizer = OCRRecognizer()
    
    line_results = []
    for i, line in enumerate(lines):
        try:
            h_line, w_line = line.shape
            text = recognizer.predict_line(line)
            
            if text and len(text.strip()) > 0 and recognizer._is_valid_text(text):
                print(f"✓ Line {i} ({w_line}x{h_line}): '{text}'")
                line_results.append(text)
            else:
                print(f"✗ Line {i} ({w_line}x{h_line}): [No valid text]")
        except Exception as e:
            print(f"✗ Line {i}: Error - {e}")
    
    # ==========================================
    # 6. FULL PAGE OCR TEST
    # ==========================================
    print("\nFull Page OCR Test")
    print("=" * 70)
    
    text = recognizer.predict(image_path, debug=debug)
    
    if text.startswith("[NO TEXT"):
        print(f"✗ {text}")
        print("\nRecommendations:")
        print("1. Verify image contains readable text")
        print("2. Check debug images in debug_output/ directory")
        print("3. Ensure text is not too small or blurry")
        return False
    
    print("Recognized Text:")
    print("-" * 70)
    print(text)
    print("-" * 70)
    
    # ==========================================
    # 7. STATISTICS
    # ==========================================
    lines_recognized = len(line_results)
    total_chars = sum(len(line) for line in line_results)
    avg_chars = total_chars / len(line_results) if line_results else 0
    
    print("\nStatistics:")
    print(f"  Total lines: {lines_recognized}")
    print(f"  Total characters: {total_chars}")
    print(f"  Avg chars/line: {avg_chars:.1f}")
    
    print("\n✓ Recognition successful")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why OCR fails on specific image"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1
    
    success = diagnose_image(args.image, debug=args.debug)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
