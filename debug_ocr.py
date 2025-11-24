#!/usr/bin/env python3
"""
Debug OCR prediction - See exactly where it fails
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import torch
import numpy as np
from preprocessing.recognize import OCRRecognizer
from preprocessing.line_segment import segment_lines


def debug_ocr_full_pipeline(image_path):
    """
    Debug full OCR pipeline step by step
    """
    print(f"\n{'='*70}")
    print(f"OCR FULL PIPELINE DEBUG")
    print(f"{'='*70}\n")
    
    print(f"Image: {Path(image_path).name}\n")
    
    # Step 1: Load image
    print(f"[1/4] Loading image...")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Failed to load image")
        return
    
    h, w = img.shape
    print(f"✓ Loaded: {w}×{h} pixels\n")
    
    # Step 2: Line segmentation
    print(f"[2/4] Segmenting lines...")
    lines = segment_lines(str(image_path), debug=False)
    print(f"✓ Found {len(lines)} lines\n")
    
    if len(lines) == 0:
        print("❌ No lines found - cannot proceed")
        return
    
    for i, line in enumerate(lines[:3], 1):
        h_l, w_l = line.shape
        print(f"  Line {i}: {w_l}×{h_l} pixels, intensity mean={np.mean(line):.1f}")
    if len(lines) > 3:
        print(f"  ... and {len(lines)-3} more")
    
    # Step 3: Initialize model
    print(f"\n[3/4] Loading model...")
    try:
        recognizer = OCRRecognizer()
        print(f"✓ Model loaded successfully")
        print(f"  Device: {next(recognizer.model.parameters()).device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 4: Test on each line
    print(f"\n[4/4] Predicting text...")
    
    all_text = []
    for i, line in enumerate(lines, 1):
        try:
            # Try to predict on this line
            result = recognizer.predict_line(line)
            
            if result and result != "[NO TEXT DETECTED - RECOGNITION FAILED]":
                print(f"  Line {i}: ✓ {result[:60]}...")
                all_text.append(result)
            else:
                print(f"  Line {i}: ✗ [No text detected]")
        
        except Exception as e:
            print(f"  Line {i}: ✗ Error: {str(e)[:50]}")
    
    print(f"\n{'─'*70}")
    print(f"RESULTS: {len(all_text)}/{len(lines)} lines recognized")
    print(f"{'─'*70}\n")
    
    if all_text:
        print("Full extracted text:")
        print("\n".join(all_text))
    else:
        print("❌ NO TEXT EXTRACTED - Check line quality or model weights")
    
    return all_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_images = list(Path("datasets/printed_notes_extracted").glob("*.png"))[:1]
        if not test_images:
            print("Usage: python debug_ocr.py <image_path>")
            sys.exit(1)
        image_path = test_images[0]
    else:
        image_path = sys.argv[1]
    
    debug_ocr_full_pipeline(str(image_path))
