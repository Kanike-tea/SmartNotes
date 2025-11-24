#!/usr/bin/env python3
"""
SmartNotes Production OCR - College Project Edition
Uses multiple fallback strategies to ensure working output
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports
setup_imports()

import cv2
import torch
import numpy as np
from preprocessing.line_segment import segment_lines


class ProductionOCR:
    """Production-ready OCR with multiple fallback strategies"""
    
    def __init__(self):
        """Initialize with all available OCR methods"""
        self.use_model = True
        self.use_tesseract = False
        self.use_paddle = False
        
        # Try importing optional dependencies
        try:
            import pytesseract
            self.use_tesseract = True
            print("✓ Tesseract OCR available")
        except:
            pass
        
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            self.use_paddle = True
            print("✓ PaddleOCR available")
        except:
            pass
        
        # Initialize SmartNotes model
        try:
            from preprocessing.recognize import OCRRecognizer
            self.recognizer = OCRRecognizer()
            print("✓ SmartNotes model loaded")
        except Exception as e:
            print(f"✗ SmartNotes model failed: {e}")
            self.use_model = False
    
    def ocr_image(self, image_path):
        """
        Run OCR with multiple fallback strategies
        """
        results = {}
        
        # Strategy 1: SmartNotes model
        if self.use_model:
            try:
                result = self._smartnotes_ocr(image_path)
                if result:
                    results['smartnotes'] = result
            except Exception as e:
                pass
        
        # Strategy 2: PaddleOCR (if available)
        if self.use_paddle:
            try:
                result = self._paddle_ocr(image_path)
                if result:
                    results['paddle'] = result
            except Exception as e:
                pass
        
        # Strategy 3: Tesseract (if available)
        if self.use_tesseract:
            try:
                result = self._tesseract_ocr(image_path)
                if result:
                    results['tesseract'] = result
            except Exception as e:
                pass
        
        return results
    
    def _smartnotes_ocr(self, image_path):
        """Use SmartNotes model"""
        lines = segment_lines(str(image_path), debug=False)
        if not lines:
            return None
        
        text = []
        for line in lines:
            try:
                # Test both normal and inverted
                result_normal = self.recognizer.predict_line(line)
                result_inv = self.recognizer.predict_line(255 - line)
                
                # Pick the longer result
                result = result_normal if len(result_normal) >= len(result_inv) else result_inv
                
                if result and result != "[NO TEXT DETECTED - RECOGNITION FAILED]":
                    text.append(result)
            except:
                pass
        
        return "\n".join(text) if text else None
    
    def _paddle_ocr(self, image_path):
        """Use PaddleOCR"""
        try:
            result = self.paddle_ocr.ocr(str(image_path), cls=True)
            text = []
            for line in result:
                for item in line:
                    text.append(item[1])  # Get text
            return "\n".join(text) if text else None
        except:
            return None
    
    def _tesseract_ocr(self, image_path):
        """Use Tesseract"""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text if text.strip() else None
        except:
            return None


def demo():
    """Demo the production OCR"""
    print(f"\n{'='*70}")
    print(f"SmartNotes Production OCR")
    print(f"College Project Edition")
    print(f"{'='*70}\n")
    
    # Initialize
    ocr = ProductionOCR()
    
    # Find test images
    test_images = list(Path("datasets/printed_notes_extracted").rglob("*.png"))[:3]
    
    if not test_images:
        print("No test images found")
        return
    
    print(f"Found {len(test_images)} test images\n")
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"[{idx}] {image_path.name}")
        
        results = ocr.ocr_image(str(image_path))
        
        if not results:
            print("  ✗ No OCR method produced output\n")
            continue
        
        for method, text in results.items():
            lines = len(text.split('\n'))
            preview = text.split('\n')[0][:60] if text else "[empty]"
            print(f"  {method:12} ({lines} lines): {preview}...")
        
        print()


if __name__ == "__main__":
    demo()
