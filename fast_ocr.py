#!/usr/bin/env python3
"""
SmartNotes Fast OCR - College Project Demo
Quick script to test OCR on your images
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import torch
from preprocessing.recognize import OCRRecognizer


def fast_ocr(image_path, verbose=False):
    """
    Fast OCR on a single image
    
    Args:
        image_path: Path to image file
        verbose: Print detailed info
    """
    print(f"\n{'='*70}")
    print(f"SmartNotes Fast OCR")
    print(f"{'='*70}\n")
    
    try:
        # Load image
        print(f"[1/3] Loading image: {image_path}")
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Error: Could not load image")
            return False
        h, w = img.shape
        print(f"✓ Image loaded: {w}×{h}")
        
        # Initialize recognizer
        print(f"\n[2/3] Loading OCR model...")
        recognizer = OCRRecognizer()
        print(f"✓ Model loaded")
        
        # Run OCR
        print(f"\n[3/3] Running OCR...")
        result = recognizer.predict(str(image_path), debug=verbose)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(result)
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fast_ocr.py <image_path> [--verbose]")
        print("\nExample:")
        print("  python fast_ocr.py lab_manual.png")
        print("  python fast_ocr.py notes.jpg --verbose")
        sys.exit(1)
    
    image_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "--debug" in sys.argv
    
    success = fast_ocr(image_path, verbose=verbose)
    sys.exit(0 if success else 1)
