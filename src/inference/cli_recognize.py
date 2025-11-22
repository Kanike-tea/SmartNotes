#!/usr/bin/env python3
"""
SmartNotes OCR CLI - Production inference tool.

Recognize text from handwritten or printed images using epoch 6 OCR model + ARPA LM.

Usage:
    python3 cli_recognize.py --image path/to/image.png
    python3 cli_recognize.py --image path/to/image.png --use-lm
    python3 cli_recognize.py --batch input_dir/ --output results.txt
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.recognize import OCRLMInference
from utils import get_logger
import cv2
import torch
import numpy as np

logger = get_logger(__name__)


class OCRCLITool:
    """CLI tool for OCR inference."""
    
    def __init__(self, use_lm: bool = True, device: Optional[str] = None):
        """Initialize CLI tool."""
        print("Initializing OCR engine...")
        self.inference = OCRLMInference(use_lm=use_lm)
        print(f"✓ Ready to recognize text\n")
    
    def recognize_image(self, image_path: str) -> Dict[str, Any]:
        """
        Recognize text from single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with prediction and metadata
        """
        image_path_obj = Path(image_path)
        
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        img = cv2.imread(str(image_path_obj), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to model input size
        img = cv2.resize(img, (128, 32))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        # Run inference
        results = self.inference.infer(img_tensor)
        pred_text = results[0][0] if results else ""
        
        return {
            "image": str(image_path),
            "prediction": pred_text,
            "confidence": "high" if len(pred_text) > 0 else "low",
            "model": "epoch_6_crnn",
            "lm_enabled": self.inference.use_lm
        }
    
    def recognize_batch(
        self,
        input_dir: str,
        output_file: Optional[str] = None,
        image_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recognize text from all images in directory.
        
        Args:
            input_dir: Directory containing images
            output_file: Optional file to save results as JSONL
            image_extensions: Image file extensions to process
            
        Returns:
            List of results
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []
        
        logger.info(f"Processing {len(image_files)} images...")
        
        results = []
        from tqdm import tqdm
        
        for img_path in tqdm(image_files, desc="Recognizing"):
            try:
                result = self.recognize_image(str(img_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({
                    "image": str(img_path),
                    "error": str(e)
                })
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            logger.info(f"Results saved to: {output_path}")
        
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmartNotes OCR - Handwriting Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recognize single image
  python3 cli_recognize.py --image document.png
  
  # Recognize with language model
  python3 cli_recognize.py --image document.png --use-lm
  
  # Batch process directory
  python3 cli_recognize.py --batch ./images/ --output results.jsonl
  
  # Display results
  python3 cli_recognize.py --batch ./images/ --output results.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image for recognition"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Directory containing images for batch processing"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for batch results (JSONL format)"
    )
    parser.add_argument(
        "--use-lm",
        action="store_true",
        default=True,
        help="Use language model for decoding (default: True)"
    )
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help="Disable language model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        parser.print_help()
        sys.exit(1)
    
    use_lm = args.use_lm and not args.no_lm
    
    # Initialize tool
    tool = OCRCLITool(use_lm=use_lm, device=args.device)
    
    try:
        if args.image:
            # Single image
            result = tool.recognize_image(args.image)
            
            print("\n" + "="*70)
            print("RECOGNITION RESULT")
            print("="*70)
            print(f"Image: {result['image']}")
            print(f"Text: {result['prediction']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Model: {result['model']}")
            print(f"LM: {'Enabled' if result['lm_enabled'] else 'Disabled'}")
            print("="*70 + "\n")
        
        elif args.batch:
            # Batch processing
            results = tool.recognize_batch(
                args.batch,
                output_file=args.output
            )
            
            if args.verbose:
                print("\n" + "="*70)
                print("BATCH RESULTS")
                print("="*70)
                for result in results[:10]:  # Show first 10
                    if "error" not in result:
                        print(f"{Path(result['image']).name}: {result['prediction']}")
                if len(results) > 10:
                    print(f"... and {len(results) - 10} more")
                print("="*70 + "\n")
            
            print(f"✓ Processed {len(results)} images")
            if args.output:
                print(f"✓ Results saved to: {args.output}\n")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
