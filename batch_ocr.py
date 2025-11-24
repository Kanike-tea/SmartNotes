#!/usr/bin/env python3
"""
SmartNotes Batch OCR - Optimized for College Project
Handles large images, multiple pages, and produces best accuracy
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import torch
import numpy as np
from preprocessing.recognize import OCRRecognizer


def preprocess_large_image(image_path):
    """
    Preprocess large images by downsampling if needed
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # If too large, downsample to prevent memory issues
    h, w = img.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"   Downsampled from {w}×{h} to {new_w}×{new_h}")
    
    return img


def batch_ocr(image_paths, verbose=False):
    """
    Run OCR on multiple images
    
    Args:
        image_paths: List of image paths or single path
        verbose: Print verbose output
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    print(f"\n{'='*70}")
    print(f"SmartNotes Batch OCR - College Project Edition")
    print(f"{'='*70}\n")
    
    # Initialize recognizer once
    print(f"Loading OCR model...")
    recognizer = OCRRecognizer()
    print(f"✓ Model ready\n")
    
    results = {}
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {Path(image_path).name}")
        
        try:
            # Preprocess
            print(f"  Loading image...")
            img = preprocess_large_image(image_path)
            if img is None:
                print(f"  ❌ Failed to load image")
                results[image_path] = "[ERROR - Could not load image]"
                continue
            
            h, w = img.shape
            print(f"  Image size: {w}×{h}")
            
            # Run OCR
            print(f"  Running OCR...")
            result = recognizer.predict(str(image_path), debug=False)
            
            results[image_path] = result
            
            # Quick preview
            lines = result.split('\n')
            preview = lines[0][:80] if lines else "[No text]"
            print(f"  ✓ Success ({len(lines)} lines): {preview}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[image_path] = f"[ERROR - {str(e)[:50]}]"
    
    # Display all results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}\n")
    
    for image_path, result in results.items():
        print(f"File: {Path(image_path).name}")
        print(f"{'-'*70}")
        print(result)
        print(f"\n")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_ocr.py <image1> [image2] [image3] ... [--verbose]")
        print("\nExample:")
        print("  python batch_ocr.py lab_manual.png")
        print("  python batch_ocr.py page1.jpg page2.jpg page3.jpg")
        print("  python batch_ocr.py *.png --verbose")
        sys.exit(1)
    
    # Extract image paths (all args except --verbose flags)
    image_paths = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    verbose = "--verbose" in sys.argv or "--debug" in sys.argv
    
    if not image_paths:
        print("No image paths provided")
        sys.exit(1)
    
    results = batch_ocr(image_paths, verbose=verbose)
