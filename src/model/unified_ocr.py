#!/usr/bin/env python3
"""
unified_ocr.py - Complete OCR Engine for SmartNotes

Combines:
- Pytesseract (for printed text)
- CRNN (for handwriting)  
- Batch processing
- Line segmentation fallback

This is the ONLY file you need for all OCR tasks.
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

from smartnotes.paths import setup_imports, get_checkpoint_dir
setup_imports()

import torch
import cv2
import numpy as np
from PIL import Image

# OCR Engines
import pytesseract
from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer

# Optional line segmentation
try:
    from preprocessing.line_segment import segment_lines
    HAS_LINE_SEGMENT = True
except ImportError:
    HAS_LINE_SEGMENT = False
    print("[INFO] line_segment not available - full page mode only")


class UnifiedOCREngine:
    """
    Unified OCR Engine supporting multiple modes:
    - Pytesseract (primary for printed text)
    - CRNN (fallback for handwriting)
    - Line-by-line processing
    - Batch processing
    """
    
    def __init__(self, 
                 checkpoint_path="checkpoints/ocr_epoch_6.pth",
                 prefer_tesseract=True,
                 confidence_threshold=30):
        """
        Initialize OCR Engine
        
        Args:
            checkpoint_path: Path to CRNN model checkpoint
            prefer_tesseract: If True, try Tesseract first (recommended)
            confidence_threshold: Minimum Tesseract confidence (0-100)
        """
        self.prefer_tesseract = prefer_tesseract
        self.confidence_threshold = confidence_threshold
        
        # Setup CRNN
        self.device = torch.device("cpu")
        self.tokenizer = TextTokenizer()
        self.num_classes = len(self.tokenizer.chars)
        self.model = CRNN(num_classes=self.num_classes).to(self.device)
        
        # Load checkpoint if available
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.has_crnn = True
            print(f"[OK] ✓ CRNN loaded from {checkpoint_path}")
        else:
            self.has_crnn = False
            print(f"[WARNING] CRNN checkpoint not found: {checkpoint_path}")
        
        # Check Tesseract
        try:
            pytesseract.get_tesseract_version()
            self.has_tesseract = True
            print("[OK] ✓ Pytesseract available")
        except Exception as e:
            self.has_tesseract = False
            print(f"[WARNING] Pytesseract not available: {e}")
    
    # ========================================================================
    # TESSERACT ENGINE
    # ========================================================================
    
    def recognize_with_tesseract(self, image_path, psm=3, debug=False):
        """
        Use Pytesseract to extract text (best for printed)
        
        Args:
            image_path: Path to image or PIL Image object
            psm: Page segmentation mode (3=auto, 6=uniform block, 11=sparse)
            debug: Print debug info
            
        Returns:
            tuple: (text, confidence)
        """
        if not self.has_tesseract:
            return None, 0
        
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                img = Image.open(image_path)
            else:
                img = image_path  # Already PIL Image
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Try multiple PSM modes
            best_text = ""
            best_conf = 0
            
            psm_modes = [psm, 3, 6, 11] if psm != 3 else [3, 6, 11]
            
            for mode in psm_modes:
                config = f'--psm {mode} --oem 3'
                
                # Get text with confidence data
                data = pytesseract.image_to_data(
                    img, 
                    config=config, 
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract valid text
                texts = []
                confidences = []
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        texts.append(text)
                        conf = int(data['conf'][i])
                        if conf > 0:
                            confidences.append(conf)
                
                if texts and confidences:
                    full_text = ' '.join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    
                    if debug:
                        print(f"[TESSERACT] PSM {mode}: {len(texts)} words, conf={avg_conf:.1f}")
                    
                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_text = full_text
            
            if debug and best_text:
                print(f"[TESSERACT] Best result: conf={best_conf:.1f}, {len(best_text)} chars")
            
            return best_text, best_conf
        
        except Exception as e:
            if debug:
                print(f"[ERROR] Tesseract failed: {e}")
            return None, 0
    
    # ========================================================================
    # CRNN ENGINE
    # ========================================================================
    
    def preprocess_line(self, img):
        """Preprocess line image for CRNN"""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            h, w = img.shape
            
            # Resize to intermediate height
            target_height = 64
            scale = target_height / h
            new_width = max(20, int(w * scale))
            img_resized = cv2.resize(img, (new_width, target_height))
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_resized)
            
            # Sharpen
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Binarize
            binary = cv2.adaptiveThreshold(
                sharpened, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Final resize
            final = cv2.resize(binary, (128, 32))
            
            # Normalize and tensorize
            normalized = final.astype(np.float32) / 255.0
            tensor = torch.tensor(normalized).float().unsqueeze(0).unsqueeze(0)
            
            return tensor.to(self.device)
        
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return None
    
    def predict_line(self, line_image):
        """Predict text from line using CRNN"""
        if not self.has_crnn:
            return ""
        
        try:
            if isinstance(line_image, np.ndarray):
                img_array = line_image
            else:
                img_array = np.array(line_image)
            
            tensor = self.preprocess_line(img_array)
            if tensor is None:
                return ""
            
            with torch.no_grad():
                logits = self.model(tensor)
            
            # Decode
            pred_indices = torch.argmax(logits, dim=2)[0].cpu().numpy()
            
            text = ""
            prev_idx = -1
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:
                    if idx - 1 < len(self.tokenizer.chars):
                        text += self.tokenizer.chars[idx - 1]
                prev_idx = idx
            
            return text.strip()
        
        except Exception as e:
            print(f"[ERROR] CRNN prediction failed: {e}")
            return ""
    
    def recognize_with_crnn(self, image_path, debug=False):
        """
        Use CRNN to extract text (best for handwriting)
        
        Args:
            image_path: Path to image
            debug: Print debug info
            
        Returns:
            tuple: (text, confidence)
        """
        if not self.has_crnn or not HAS_LINE_SEGMENT:
            return None, 0
        
        try:
            # Segment into lines
            lines = segment_lines(image_path, debug=debug)
            
            if not lines:
                return None, 0
            
            results = []
            for i, line in enumerate(lines):
                if line is None or line.size == 0:
                    continue
                
                h, w = line.shape
                if h < 10 or w < 20:
                    continue
                
                text = self.predict_line(line)
                
                if text and len(text.strip()) > 2:
                    results.append(text)
                    if debug:
                        print(f"[CRNN] Line {i}: {text}")
            
            if results:
                full_text = '\n'.join(results)
                # Rough confidence based on output quality
                confidence = min(95, len(results) * 10)
                return full_text, confidence
            
            return None, 0
        
        except Exception as e:
            if debug:
                print(f"[ERROR] CRNN failed: {e}")
            return None, 0
    
    # ========================================================================
    # UNIFIED INTERFACE
    # ========================================================================
    
    def recognize(self, image_path, mode='auto', debug=False):
        """
        Main OCR method - automatically selects best engine
        
        Args:
            image_path: Path to image file or PIL Image
            mode: 'auto', 'tesseract', 'crnn'
            debug: Print debug information
            
        Returns:
            dict: {
                'text': extracted text,
                'confidence': confidence score (0-100),
                'engine': engine used ('tesseract' or 'crnn'),
                'success': True/False
            }
        """
        result = {
            'text': '',
            'confidence': 0,
            'engine': None,
            'success': False
        }
        
        if debug:
            print(f"\n[OCR] Processing: {image_path}")
            print(f"[OCR] Mode: {mode}")
        
        # Auto mode: Try Tesseract first, fallback to CRNN
        if mode == 'auto':
            if self.prefer_tesseract and self.has_tesseract:
                text, conf = self.recognize_with_tesseract(image_path, debug=debug)
                
                if text and len(text.strip()) > 20 and conf > self.confidence_threshold:
                    result['text'] = text
                    result['confidence'] = conf
                    result['engine'] = 'tesseract'
                    result['success'] = True
                    
                    if debug:
                        print(f"[OK] ✓ Tesseract succeeded (conf={conf:.1f})")
                    
                    return result
                
                if debug:
                    print(f"[INFO] Tesseract insufficient (conf={conf:.1f}), trying CRNN...")
            
            # Fallback to CRNN
            if self.has_crnn:
                text, conf = self.recognize_with_crnn(image_path, debug=debug)
                
                if text:
                    result['text'] = text
                    result['confidence'] = conf
                    result['engine'] = 'crnn'
                    result['success'] = True
                    
                    if debug:
                        print(f"[OK] ✓ CRNN succeeded")
                    
                    return result
        
        # Force Tesseract
        elif mode == 'tesseract':
            if self.has_tesseract:
                text, conf = self.recognize_with_tesseract(image_path, debug=debug)
                if text:
                    result['text'] = text
                    result['confidence'] = conf
                    result['engine'] = 'tesseract'
                    result['success'] = True
                    return result
        
        # Force CRNN
        elif mode == 'crnn':
            if self.has_crnn:
                text, conf = self.recognize_with_crnn(image_path, debug=debug)
                if text:
                    result['text'] = text
                    result['confidence'] = conf
                    result['engine'] = 'crnn'
                    result['success'] = True
                    return result
        
        # Both failed
        if debug:
            print("[ERROR] All OCR engines failed")
        
        result['text'] = "[NO TEXT DETECTED - RECOGNITION FAILED]"
        return result
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def process_batch(self, directory, pattern="*.png", output_dir="ocr_results", 
                     mode='auto', debug=False):
        """
        Process multiple images and save results
        
        Args:
            directory: Directory containing images
            pattern: File pattern (e.g., "*.png", "*.jpg")
            output_dir: Output directory for results
            mode: OCR mode ('auto', 'tesseract', 'crnn')
            debug: Print debug info
            
        Returns:
            dict: Statistics about processing
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_files = sorted(Path(directory).rglob(pattern))
        
        stats = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "tesseract_count": 0,
            "crnn_count": 0,
            "total_text_length": 0,
            "avg_confidence": 0,
            "start_time": datetime.now().isoformat()
        }
        
        print(f"\n{'='*70}")
        print(f"Batch OCR Processing")
        print(f"{'='*70}\n")
        print(f"Found: {len(image_files)} images")
        print(f"Mode: {mode}")
        print(f"Output: {output_path.absolute()}\n")
        
        results = {}
        confidences = []
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx:3d}/{len(image_files)}] {image_path.name:50s} ... ", 
                  end="", flush=True)
            
            # Process image
            result = self.recognize(str(image_path), mode=mode, debug=False)
            
            # Save to file
            output_file = output_path / f"{image_path.stem}.txt"
            with open(output_file, 'w') as f:
                f.write(result['text'])
            
            # Update stats
            if result['success']:
                stats["successful"] += 1
                stats["total_text_length"] += len(result['text'])
                
                if result['engine'] == 'tesseract':
                    stats["tesseract_count"] += 1
                elif result['engine'] == 'crnn':
                    stats["crnn_count"] += 1
                
                confidences.append(result['confidence'])
                
                status = f"✓ {result['engine']:10s} ({len(result['text']):5d} chars, conf={result['confidence']:.1f})"
            else:
                stats["failed"] += 1
                status = "✗ failed"
            
            print(status)
            results[str(image_path)] = result
        
        # Calculate averages
        if confidences:
            stats["avg_confidence"] = sum(confidences) / len(confidences)
        
        stats["end_time"] = datetime.now().isoformat()
        
        # Save summary
        summary_file = output_path / "SUMMARY.json"
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create combined file
        combined_file = output_path / "ALL_TEXT.txt"
        with open(combined_file, 'w') as f:
            for image_path in sorted(results.keys()):
                f.write(f"\n{'='*70}\n")
                f.write(f"File: {Path(image_path).name}\n")
                f.write(f"Engine: {results[image_path]['engine']}\n")
                f.write(f"Confidence: {results[image_path]['confidence']:.1f}\n")
                f.write(f"{'='*70}\n\n")
                f.write(results[image_path]['text'])
                f.write(f"\n\n")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}\n")
        print(f"Total images: {stats['total_images']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Tesseract: {stats['tesseract_count']}")
        print(f"CRNN: {stats['crnn_count']}")
        print(f"Total text: {stats['total_text_length']:,} characters")
        print(f"Avg confidence: {stats['avg_confidence']:.1f}")
        print(f"\n✓ Results saved to: {output_path.absolute()}\n")
        
        return stats


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Global instance for simple API
_engine = None

def recognize_image(image_path, debug=False):
    """Simple function for backward compatibility with old code"""
    global _engine
    if _engine is None:
        _engine = UnifiedOCREngine()
    
    result = _engine.recognize(image_path, mode='auto', debug=debug)
    return result['text']


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified OCR Engine for SmartNotes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python unified_ocr.py --image document.png
  
  # Batch processing
  python unified_ocr.py --batch datasets/printed_notes_extracted
  
  # Force Tesseract only
  python unified_ocr.py --image doc.png --mode tesseract
  
  # Debug mode
  python unified_ocr.py --image doc.png --debug
        """
    )
    
    parser.add_argument('--image', type=str, help='Process single image')
    parser.add_argument('--batch', type=str, help='Process directory of images')
    parser.add_argument('--pattern', default='*.png', help='File pattern for batch (default: *.png)')
    parser.add_argument('--output', default='ocr_results', help='Output directory (default: ocr_results)')
    parser.add_argument('--mode', choices=['auto', 'tesseract', 'crnn'], 
                       default='auto', help='OCR mode (default: auto)')
    parser.add_argument('--checkpoint', default='checkpoints/ocr_epoch_6.pth',
                       help='CRNN checkpoint path')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.print_help()
        sys.exit(1)
    
    # Initialize engine
    engine = UnifiedOCREngine(
        checkpoint_path=args.checkpoint,
        prefer_tesseract=True
    )
    
    # Single image
    if args.image:
        result = engine.recognize(args.image, mode=args.mode, debug=args.debug)
        
        print(f"\n{'='*70}")
        print(f"OCR RESULT")
        print(f"{'='*70}\n")
        print(f"Engine: {result['engine']}")
        print(f"Confidence: {result['confidence']:.1f}")
        print(f"Success: {result['success']}")
        print(f"\n{'-'*70}\n")
        print(result['text'])
        print(f"\n{'='*70}\n")
    
    # Batch processing
    if args.batch:
        engine.process_batch(
            directory=args.batch,
            pattern=args.pattern,
            output_dir=args.output,
            mode=args.mode,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
