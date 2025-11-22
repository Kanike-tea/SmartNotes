"""
Test/Inference script for SmartNotes OCR model.

Runs inference on test samples and displays predictions with metrics.
"""

import sys
import pytest
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from utils import get_logger, get_device, validate_checkpoint_path, calculate_cer, calculate_wer
from src.dataloader.ocr_dataloader import SmartNotesOCRDataset
from src.model.ocr_model import CRNN

logger = get_logger(__name__)
 
pytestmark = pytest.mark.skip(reason="Script-style inference runner — not a unit test")


class OCRInference:
    """
    Inference engine for OCR model.
    
    Handles model loading, inference, and metrics calculation.
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: torch.device = None,
        use_cpu: bool = False
    ) -> None:
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            use_cpu: Force CPU usage
        """
        if checkpoint_path is None:
            checkpoint_path = Config.inference.CHECKPOINT_PATH
        
        if device is None:
            device = get_device(force_cpu=use_cpu or Config.inference.USE_CPU)
        
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Load model
        self._load_model(checkpoint_path)
        logger.info(f"Inference engine initialized on {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        if not validate_checkpoint_path(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Initializing model with random weights")
            from src.dataloader.ocr_dataloader import TextTokenizer
            self.tokenizer = TextTokenizer()
            num_classes = len(self.tokenizer.chars)
            self.model = CRNN(num_classes=num_classes).to(self.device)
            return
        
        try:
            # Load dataset to get tokenizer and num_classes
            dataset = SmartNotesOCRDataset(mode='val')
            self.tokenizer = dataset.tokenizer
            num_classes = len(self.tokenizer.chars)
            
            # Create model
            self.model = CRNN(num_classes=num_classes).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle both full checkpoint and state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model state dict")
            
            self.model.eval()
            logger.info(f"Model loaded successfully from {checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def predict(self, image: torch.Tensor) -> str:
        """
        Run inference on a single image.
        
        Args:
            image: Image tensor (1, 32, 128) or (1, 1, 32, 128)
            
        Returns:
            Predicted text string
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            preds = self.model(image)
            preds = preds.permute(1, 0, 2).cpu()
            probs = torch.softmax(preds, dim=2)
            seq = torch.argmax(probs[0], dim=1).numpy()
        
        return self.tokenizer.decode(seq)


def test_model(
    dataset_mode: str = 'val',
    num_samples: Optional[int] = None,
    checkpoint_path: Optional[str] = None
) -> None:
    """
    Run inference on test set and display results.
    
    Args:
        dataset_mode: 'val' or 'test'
        num_samples: Number of samples to test (None for all)
        checkpoint_path: Path to checkpoint (uses config if None)
    """
    try:
        logger.info("Initializing inference engine...")
        
        # Load dataset
        logger.info(f"Loading {dataset_mode} dataset...")
        dataset = SmartNotesOCRDataset(
            mode=dataset_mode,
            max_samples=num_samples
        )
        
        # Initialize inference
        inference = OCRInference(checkpoint_path=checkpoint_path)
        
        # Run inference
        logger.info(f"\nRunning inference on {len(dataset)} samples...")
        
        num_displayed = 0
        num_skipped = 0
        total_cer = 0.0
        total_wer = 0.0
        
        for idx in range(len(dataset)):
            try:
                img, label = dataset[idx]
                
                # Skip invalid samples
                if img is None:
                    num_skipped += 1
                    continue
                
                # Run prediction
                pred_text = inference.predict(img)
                gt_text = inference.tokenizer.decode(label.numpy())
                
                # Calculate metrics
                cer = calculate_cer(pred_text, gt_text)
                wer = calculate_wer(pred_text, gt_text)
                total_cer += cer
                total_wer += wer
                
                # Display sample
                if num_displayed < Config.inference.MAX_SAMPLES_TO_DISPLAY:
                    logger.info(f"\n--- Sample {num_displayed + 1} ---")
                    logger.info(f"Predicted: {pred_text}")
                    logger.info(f"Ground Truth: {gt_text}")
                    logger.info(f"CER: {cer:.4f}, WER: {wer:.4f}")
                    num_displayed += 1
            
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                num_skipped += 1
                continue
        
        # Summary metrics
        num_processed = len(dataset) - num_skipped
        if num_processed > 0:
            avg_cer = total_cer / num_processed
            avg_wer = total_wer / num_processed
            logger.info(f"\n{'='*60}")
            logger.info(f"Summary Statistics (from {num_processed} processed samples)")
            logger.info(f"{'='*60}")
            logger.info(f"Average CER: {avg_cer:.4f}")
            logger.info(f"Average WER: {avg_wer:.4f}")
            logger.info(f"Skipped samples: {num_skipped}")
            logger.info(f"{'='*60}\n")
        else:
            logger.warning("No valid samples processed!")
    
    except Exception as e:
        logger.error(f"Fatal error during inference: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OCR inference on test set")
    parser.add_argument(
        "--mode",
        choices=['val', 'test'],
        default='val',
        help="Dataset mode to use"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to test (None for all)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    test_model(
        dataset_mode=args.mode,
        num_samples=args.num_samples,
        checkpoint_path=args.checkpoint
    )

