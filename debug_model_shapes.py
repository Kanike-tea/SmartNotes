#!/usr/bin/env python3
"""
Debug model output shapes
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


def debug_model_output_shapes(image_path):
    """See exactly what the model is outputting"""
    
    print(f"\n{'='*70}")
    print(f"MODEL OUTPUT SHAPE DEBUG")
    print(f"{'='*70}\n")
    
    # Get first line
    lines = segment_lines(str(image_path), debug=False)
    if not lines:
        print("No lines found")
        return
    
    test_line = lines[0]
    print(f"Test line shape: {test_line.shape}\n")
    
    # Initialize model
    recognizer = OCRRecognizer()
    print(f"Model num_classes: {recognizer.model.num_classes}\n")
    
    # Preprocess
    tensor = recognizer.preprocess_line(test_line)
    print(f"Preprocessed tensor shape: {tensor.shape}\n")
    
    # Forward pass
    with torch.no_grad():
        logits = recognizer.model(tensor)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: (T, B, C) where C = num_classes + 1 = {recognizer.model.num_classes + 1}\n")
    
    # Analyze
    T, B, C = logits.shape
    print(f"T (sequence length): {T}")
    print(f"B (batch size): {B}")
    print(f"C (classes): {C}")
    print(f"  = {recognizer.model.num_classes} chars + 1 blank\n")
    
    # Check tokenizer
    print(f"Tokenizer chars ({len(recognizer.tokenizer.chars)}): {recognizer.tokenizer.chars[:50]}...\n")
    
    # Manual decoding - CORRECT WAY
    print(f"{'─'*70}")
    print(f"DECODING TEST")
    print(f"{'─'*70}\n")
    
    # Get argmax - should give one index per timestep
    pred_indices = torch.argmax(logits, dim=2)  # Shape: (T, B)
    print(f"Argmax shape (T, B): {pred_indices.shape}")
    
    pred_indices_seq = pred_indices.squeeze().cpu().numpy()
    print(f"Squeezed indices shape: {pred_indices_seq.shape}")
    print(f"First 20 indices: {pred_indices_seq[:20]}\n")
    
    # Decode correctly
    text = ""
    prev_idx = -1
    for idx in pred_indices_seq:
        # idx=0 is blank, idx=1+ are characters
        if idx != 0 and idx != prev_idx:
            char_idx = idx - 1  # Convert from CTC index to char index
            if 0 <= char_idx < len(recognizer.tokenizer.chars):
                text += recognizer.tokenizer.chars[char_idx]
        prev_idx = idx
    
    print(f"Decoded text: {text}")
    print(f"Text length: {len(text)} chars\n")
    
    # Also show what the current broken version gets
    print(f"{'─'*70}")
    print(f"CHECKING CURRENT BROKEN CODE")
    print(f"{'─'*70}\n")
    
    # This is what the current code does (WRONG):
    pred_indices_broken = torch.argmax(logits, dim=2)[0].cpu().numpy()
    print(f"Current code argmax result shape: {pred_indices_broken.shape}")
    print(f"First 20 values: {pred_indices_broken[:20]}")


if __name__ == "__main__":
    test_images = list(Path("datasets/printed_notes_extracted").rglob("*.png"))
    if test_images:
        debug_model_output_shapes(str(test_images[0]))
    else:
        print("No test images")
