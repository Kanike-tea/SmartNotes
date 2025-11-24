#!/usr/bin/env python3
"""
Batch OCR Processor - Extract text from all images and save to files
Perfect for college project deliverables
"""

import sys
from pathlib import Path
import json
from datetime import datetime

from smartnotes.paths import setup_imports
setup_imports()

import pytesseract
from PIL import Image
import cv2


class BatchOCRProcessor:
    """Process multiple images and save results"""
    
    def __init__(self, output_dir="ocr_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.stats = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "total_text_length": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def process_image(self, image_path):
        """Process single image and return text"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            text = pytesseract.image_to_string(img, config=r'--psm 3 --oem 3')
            return text.strip()
        except Exception as e:
            return f"[ERROR: {e}]"
    
    def process_directory(self, directory, pattern="*.png"):
        """Process all images in directory"""
        image_files = sorted(Path(directory).rglob(pattern))
        print(f"\n{'='*70}")
        print(f"Batch OCR Processor")
        print(f"{'='*70}\n")
        print(f"Found {len(image_files)} images\n")
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx:3d}/{len(image_files)}] {image_path.name:50s} ... ", end="", flush=True)
            
            # Extract text
            text = self.process_image(str(image_path))
            
            # Save to file
            output_file = self.output_dir / f"{image_path.stem}.txt"
            with open(output_file, 'w') as f:
                f.write(text)
            
            # Update stats
            self.stats["total_images"] += 1
            if text and not text.startswith("[ERROR"):
                self.stats["successful"] += 1
                self.stats["total_text_length"] += len(text)
                status = f"✓ ({len(text)} chars)"
            else:
                self.stats["failed"] += 1
                status = "✗ (no text)"
            
            print(status)
            
            # Store in memory
            self.results[str(image_path)] = text
        
        self.stats["end_time"] = datetime.now().isoformat()
    
    def save_summary(self):
        """Save summary report"""
        summary_file = self.output_dir / "SUMMARY.json"
        
        with open(summary_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}\n")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Total images: {self.stats['total_images']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Total text: {self.stats['total_text_length']:,} characters")
        print(f"\n✓ Results saved to: {summary_file}\n")
    
    def create_combined_file(self):
        """Create single file with all extracted text"""
        combined_file = self.output_dir / "ALL_TEXT.txt"
        
        with open(combined_file, 'w') as f:
            for image_path in sorted(self.results.keys()):
                f.write(f"\n{'='*70}\n")
                f.write(f"File: {Path(image_path).name}\n")
                f.write(f"{'='*70}\n\n")
                f.write(self.results[image_path])
                f.write(f"\n\n")
        
        print(f"✓ Combined text: {combined_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process images with OCR")
    parser.add_argument("directory", help="Directory containing images")
    parser.add_argument("--pattern", default="*.png", help="File pattern (default: *.png)")
    parser.add_argument("--output", default="ocr_results", help="Output directory (default: ocr_results)")
    
    args = parser.parse_args()
    
    if not Path(args.directory).exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    processor = BatchOCRProcessor(output_dir=args.output)
    processor.process_directory(args.directory, pattern=args.pattern)
    processor.save_summary()
    processor.create_combined_file()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_ocr_processor.py <directory> [--pattern *.png] [--output results_dir]\n")
        print("Example:")
        print("  python batch_ocr_processor.py datasets/printed_notes_extracted")
        print("  python batch_ocr_processor.py datasets/ --output my_results --pattern *.jpg\n")
        sys.exit(1)
    
    main()
