#!/usr/bin/env python3
"""
Test different checkpoints to find the best one
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import torch
import cv2
import numpy as np
from preprocessing.recognize import OCRRecognizer
from preprocessing.line_segment import segment_lines


def test_checkpoints(image_path):
    """Test multiple checkpoints on the same image"""
    
    print(f"\n{'='*70}")
    print(f"TESTING CHECKPOINTS")
    print(f"{'='*70}\n")
    
    # Get segment lines once
    lines = segment_lines(str(image_path), debug=False)
    if not lines:
        print("No lines found")
        return
    
    # Use first line for testing
    test_line = lines[0]
    print(f"Testing on Line 1: {test_line.shape[1]}×{test_line.shape[0]} pixels\n")
    
    # Test checkpoints in order of likelihood
    checkpoints_to_try = [
        "checkpoints/ocr_best.pth",
        "checkpoints/ocr_final.pth",
        "checkpoints/ocr_finetuned_stage2_best.pth",
        "checkpoints/ocr_finetuned_epoch_20.pth",
        "checkpoints/ocr_epoch_15.pth",
        "checkpoints/ocr_epoch_20.pth",
        "checkpoints/ocr_epoch_6.pth",
    ]
    
    results = []
    
    for checkpoint in checkpoints_to_try:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            continue
        
        print(f"Testing: {checkpoint_path.name}")
        
        try:
            # Load with specific checkpoint
            recognizer = OCRRecognizer()
            
            # Manually load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle wrapped vs unwrapped
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
            else:
                state_dict = checkpoint_data
            
            recognizer.model.load_state_dict(state_dict)
            recognizer.model.eval()
            
            # Test prediction
            result = recognizer.predict_line(test_line)
            
            if result and result != "[NO TEXT DETECTED - RECOGNITION FAILED]":
                text_len = len(result)
                print(f"  ✓ {result[:70]}...")
                print(f"    Length: {text_len} chars\n")
                results.append((checkpoint, result, text_len))
            else:
                print(f"  ✗ No text detected\n")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:60]}\n")
    
    # Summary
    print(f"{'─'*70}")
    print(f"SUMMARY")
    print(f"{'─'*70}\n")
    
    if results:
        print(f"Found {len(results)} working checkpoints:\n")
        
        # Sort by text length (longer = better)
        results.sort(key=lambda x: x[2], reverse=True)
        
        for i, (checkpoint, text, length) in enumerate(results, 1):
            print(f"{i}. {Path(checkpoint).name}")
            print(f"   Output: {text[:80]}")
            print(f"   Length: {length} chars\n")
        
        print(f"\n🏆 BEST: {Path(results[0][0]).name}")
        print(f"   Reason: Longest output ({results[0][2]} chars)")
    else:
        print("❌ No working checkpoints found!")
        print("This suggests the model architecture or training is broken")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_images = list(Path("datasets/printed_notes_extracted").glob("*.png"))
        if test_images:
            image_path = test_images[0]
        else:
            print("No test images found")
            sys.exit(1)
    else:
        image_path = sys.argv[1]
    
    test_checkpoints(str(image_path))
