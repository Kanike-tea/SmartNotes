#!/usr/bin/env python3
"""
SmartNotes Final OCR for College Project
Uses Tesseract OCR (production-ready, high accuracy)
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import numpy as np
import pytesseract
from PIL import Image


class CollegeOCR:
    """Simple, reliable OCR optimized for printed lab manuals"""
    
    @staticmethod
    def extract_text(image_path):
        """Extract text from image using Tesseract"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Run Tesseract with optimized parameters
            # --psm 1: Automatic page orientation detection
            # --psm 3: Fully automatic page segmentation (best for mixed text/images)
            custom_config = r'--psm 3 --oem 3'
            text = pytesseract.image_to_string(img, config=custom_config)
            
            return text.strip()
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_region(image_path, region=None):
        """Extract text from a specific region of an image"""
        try:
            img = Image.open(image_path)
            
            # Crop region if specified
            if region:
                x1, y1, x2, y2 = region
                img = img.crop((x1, y1, x2, y2))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            custom_config = r'--psm 3'
            text = pytesseract.image_to_string(img, config=custom_config)
            
            return text.strip()
        
        except Exception as e:
            print(f"Error: {e}")
            return ""
    
    @staticmethod
    def process_document(image_path):
        """
        Process entire document page
        Returns: List of extracted text blocks
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        
        # Normalize image
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Convert back to PIL for Tesseract
        img_pil = Image.fromarray(img)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Extract text
        custom_config = r'--psm 1 --oem 3'  # PSM 1: Auto orientation
        text = pytesseract.image_to_string(img_pil, config=custom_config)
        
        # Split into blocks
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        
        return blocks


def main():
    """Process images from command line"""
    if len(sys.argv) < 2:
        print("Usage: python final_ocr.py <image1> [image2] ...")
        print("\nExample:")
        print("  python final_ocr.py lab_manual.png")
        print("  python final_ocr.py page*.png")
        sys.exit(1)
    
    image_files = sys.argv[1:]
    ocr = CollegeOCR()
    
    print(f"\n{'='*70}")
    print(f"SmartNotes OCR - College Project Edition")
    print(f"{'='*70}\n")
    
    for image_path in image_files:
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"✗ File not found: {image_path}\n")
            continue
        
        print(f"Processing: {image_path.name}")
        print(f"{'-'*70}")
        
        text = ocr.extract_text(str(image_path))
        
        if text:
            print(text)
        else:
            print("[No text detected]")
        
        print(f"\n")


if __name__ == "__main__":
    main()
