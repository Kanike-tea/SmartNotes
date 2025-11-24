#!/usr/bin/env python3
"""
SmartNotes Quick Test

Fast verification that OCR works on an image.
Shows 4-step progress and results.
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports

setup_imports()

import argparse


def quick_test(image_path):
    """Quick 4-step OCR test"""
    
    print("\n" + "=" * 70)
    print("SmartNotes Quick Test")
    print("=" * 70 + "\n")
    
    try:
        # Step 1: Load image
        print("[1/4] Loading image...")
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"✗ Error: Could not load image: {image_path}")
            return False
        
        h, w = img.shape
        print(f"✓ Loaded: {w}x{h}\n")
        
        # Step 2: Load model
        print("[2/4] Loading model...")
        from preprocessing.recognize import OCRRecognizer
        recognizer = OCRRecognizer()
        print("✓ Model loaded\n")
        
        # Step 3: Preprocess
        print("[3/4] Preprocessing...")
        from preprocessing.line_segment import segment_lines
        lines = segment_lines(image_path)
        print(f"✓ Preprocessed: {len(lines)} lines detected\n")
        
        # Step 4: Recognize
        print("[4/4] Recognizing text...\n")
        text = recognizer.predict(image_path)
        
        # Show result
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(text)
        print("=" * 70)
        
        # Statistics
        if not text.startswith("[NO TEXT"):
            lines_text = text.split('\n')
            total_chars = sum(len(line) for line in lines_text)
            
            print(f"\nLines recognized: {len(lines_text)}")
            print(f"Total characters: {total_chars}")
            print("\n✓ SUCCESS - Text recognized!")
            return True
        else:
            print(f"\n✗ FAILED - {text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick OCR test on an image"
    )
    parser.add_argument("image", help="Path to image file")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1
    
    success = quick_test(args.image)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
