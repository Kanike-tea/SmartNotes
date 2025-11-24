#!/usr/bin/env python3
"""
Test if images are inverted (white text on black = needs inversion)
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import numpy as np
from preprocessing.recognize import OCRRecognizer
from preprocessing.line_segment import segment_lines


def test_with_inverted_images(image_path):
    """Test OCR with both normal and inverted preprocessing"""
    
    print(f"\n{'='*70}")
    print(f"TESTING IMAGE INVERSION")
    print(f"{'='*70}\n")
    
    lines = segment_lines(str(image_path), debug=False)
    if not lines:
        print("No lines found")
        return
    
    test_line = lines[0]
    print(f"Test line shape: {test_line.shape}\n")
    
    # Initialize recognizer
    recognizer = OCRRecognizer()
    
    print(f"Testing: ocr_best.pth (best checkpoint)\n")
    
    # Load best checkpoint
    checkpoint_data = torch.load("checkpoints/ocr_best.pth", map_location='cpu')
    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['model_state_dict']
    else:
        state_dict = checkpoint_data
    
    recognizer.model.load_state_dict(state_dict)
    recognizer.model.eval()
    
    # Test 1: Normal
    print(f"[1] Normal preprocessing:")
    tensor = recognizer.preprocess_line(test_line)
    with torch.no_grad():
        logits = recognizer.model(tensor)
    indices = torch.argmax(logits, dim=2).squeeze().cpu().numpy()
    
    text_normal = ""
    prev = -1
    for idx in indices:
        if idx != 0 and idx != prev and idx - 1 < len(recognizer.tokenizer.chars):
            text_normal += recognizer.tokenizer.chars[idx - 1]
        prev = idx
    
    print(f"    Result: {text_normal or '[empty]'}\n")
    
    # Test 2: Inverted
    print(f"[2] Inverted preprocessing:")
    inverted = 255 - test_line
    tensor_inv = recognizer.preprocess_line(inverted)
    with torch.no_grad():
        logits_inv = recognizer.model(tensor_inv)
    indices_inv = torch.argmax(logits_inv, dim=2).squeeze().cpu().numpy()
    
    text_inverted = ""
    prev = -1
    for idx in indices_inv:
        if idx != 0 and idx != prev and idx - 1 < len(recognizer.tokenizer.chars):
            text_inverted += recognizer.tokenizer.chars[idx - 1]
        prev = idx
    
    print(f"    Result: {text_inverted or '[empty]'}\n")
    
    # Test 3: Check image statistics
    print(f"{'─'*70}")
    print(f"IMAGE ANALYSIS")
    print(f"{'─'*70}\n")
    
    print(f"Original line:")
    print(f"  Mean intensity: {np.mean(test_line):.1f}")
    print(f"  Min/Max: {np.min(test_line)}/{np.max(test_line)}")
    print(f"  Mostly light: {'Yes' if np.mean(test_line) > 128 else 'No'}\n")
    
    print(f"Inverted line:")
    print(f"  Mean intensity: {np.mean(inverted):.1f}")
    print(f"  Min/Max: {np.min(inverted)}/{np.max(inverted)}")
    print(f"  Mostly light: {'Yes' if np.mean(inverted) > 128 else 'No'}\n")


import torch

if __name__ == "__main__":
    test_images = list(Path("datasets/printed_notes_extracted").rglob("*.png"))
    if test_images:
        test_with_inverted_images(str(test_images[0]))
