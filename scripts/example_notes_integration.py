#!/usr/bin/env python3
"""
Example: Using handwritten and printed notes with SmartNotes OCR.

This script demonstrates:
1. Extracting PDFs to images
2. Loading the combined dataset
3. Training the OCR model
4. Evaluating on new datasets
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from src.dataloader.ocr_dataloader import SmartNotesOCRDataset
from src.dataloader.pdf_processor import process_notes_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_extract_pdfs():
    """Example 1: Extract PDFs to images."""
    logger.info("=" * 70)
    logger.info("Example 1: Extract PDFs to Images")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent
    
    # Extract handwritten notes
    logger.info("\nExtracting handwritten notes...")
    handwritten_input = base_dir / "datasets" / "handwritten notes"
    handwritten_output = base_dir / "datasets" / "handwritten_notes_extracted"
    
    if handwritten_input.exists():
        try:
            total = process_notes_directory(
                str(handwritten_input),
                str(handwritten_output),
                dpi=150,
                extract_regions=False
            )
            logger.info(f"✓ Extracted {total} images from handwritten notes")
        except Exception as e:
            logger.error(f"✗ Failed to extract handwritten notes: {e}")
    else:
        logger.warning(f"Handwritten notes directory not found: {handwritten_input}")
    
    # Extract printed notes
    logger.info("\nExtracting printed notes...")
    printed_input = base_dir / "datasets" / "printed notes"
    printed_output = base_dir / "datasets" / "printed_notes_extracted"
    
    if printed_input.exists():
        try:
            total = process_notes_directory(
                str(printed_input),
                str(printed_output),
                dpi=150,
                extract_regions=False
            )
            logger.info(f"✓ Extracted {total} images from printed notes")
        except Exception as e:
            logger.error(f"✗ Failed to extract printed notes: {e}")
    else:
        logger.warning(f"Printed notes directory not found: {printed_input}")


def example_2_load_dataset():
    """Example 2: Load the combined dataset."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Load Combined Dataset")
    logger.info("=" * 70)
    
    try:
        # Load training dataset
        logger.info("\nLoading training dataset...")
        train_dataset = SmartNotesOCRDataset(
            root_dir="datasets",
            mode='train',
            split_ratio=0.85
        )
        logger.info(f"✓ Loaded {len(train_dataset)} training samples")
        
        # Load validation dataset
        logger.info("\nLoading validation dataset...")
        val_dataset = SmartNotesOCRDataset(
            root_dir="datasets",
            mode='val',
            split_ratio=0.85
        )
        logger.info(f"✓ Loaded {len(val_dataset)} validation samples")
        
        # Show sample information
        logger.info("\nSample information:")
        for i in range(min(3, len(train_dataset))):
            img, label = train_dataset[i]
            logger.info(f"  Sample {i}: image shape {img.shape}, label length {len(label)}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")


def example_3_dataset_statistics():
    """Example 3: Get dataset statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Dataset Statistics")
    logger.info("=" * 70)
    
    try:
        # Load all data without max_samples limit to get total count
        dataset = SmartNotesOCRDataset(
            root_dir="datasets",
            mode='train',
            split_ratio=0.85
        )
        
        logger.info(f"\nTotal training samples: {len(dataset)}")
        
        # Sample text lengths
        text_lengths = []
        for i in range(min(100, len(dataset))):
            _, label = dataset[i]
            text_lengths.append(len(label))
        
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            max_length = max(text_lengths)
            min_length = min(text_lengths)
            
            logger.info(f"Text length statistics (from {len(text_lengths)} samples):")
            logger.info(f"  Average: {avg_length:.1f}")
            logger.info(f"  Minimum: {min_length}")
            logger.info(f"  Maximum: {max_length}")
    
    except Exception as e:
        logger.error(f"✗ Failed to get statistics: {e}")


def example_4_training_with_notes():
    """Example 4: Train with handwritten and printed notes."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Training with Notes")
    logger.info("=" * 70)
    
    logger.info("""
This shows how to train using the combined dataset that includes
handwritten and printed notes.

Step 1: Extract PDFs (shown in Example 1)
Step 2: Load dataset (shown in Example 2)
Step 3: Train using standard training script

To train, run:
    python src/training/train_ocr.py

The training script will automatically load:
    - IAM Handwriting Database
    - CensusHWR Dataset
    - GNHK Dataset
    - Handwritten notes (if extracted)
    - Printed notes (if extracted)

Monitor training progress:
    tail -f smartnotes.log
    """)


def example_5_inference_on_extracted():
    """Example 5: Run inference on extracted note images."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Inference on Extracted Notes")
    logger.info("=" * 70)
    
    logger.info("""
To run inference on extracted note images:

    python src/inference/test_ocr.py \\
        --mode val \\
        --checkpoint checkpoints/ocr_best.pth \\
        --num-samples 100

This will:
    1. Load the trained OCR model
    2. Run inference on validation set (includes extracted notes)
    3. Calculate CER (Character Error Rate) and WER (Word Error Rate)
    4. Show sample predictions

You can also run inference on a specific image:

    python -c "
    from src.inference.test_ocr import OCRInference
    
    inference = OCRInference(checkpoint='checkpoints/ocr_best.pth')
    text = inference.predict('datasets/handwritten_notes_extracted/ada/page000.png')
    print(f'Recognized text: {text}')
    "
    """)


def main():
    """Run all examples."""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "SmartNotes: Handwritten & Printed Notes Integration Examples".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    
    # Run examples
    try:
        # Example 1: Extract PDFs
        user_input = input("\nRun Example 1 (Extract PDFs)? (y/n): ").strip().lower()
        if user_input == 'y':
            example_1_extract_pdfs()
        
        # Example 2: Load dataset
        user_input = input("\nRun Example 2 (Load Dataset)? (y/n): ").strip().lower()
        if user_input == 'y':
            example_2_load_dataset()
        
        # Example 3: Statistics
        user_input = input("\nRun Example 3 (Dataset Statistics)? (y/n): ").strip().lower()
        if user_input == 'y':
            example_3_dataset_statistics()
        
        # Example 4: Training
        example_4_training_with_notes()
        
        # Example 5: Inference
        example_5_inference_on_extracted()
        
        logger.info("\n" + "=" * 70)
        logger.info("Examples Complete!")
        logger.info("=" * 70)
        logger.info("""
Next steps:
1. Review NOTES_INTEGRATION_GUIDE.md for detailed documentation
2. Run: python setup_notes_integration.py (for interactive setup)
3. Train: python src/training/train_ocr.py
4. Evaluate: python src/inference/test_ocr.py

For more help:
- NOTES_INTEGRATION_SUMMARY.md (summary of changes)
- NOTES_INTEGRATION_GUIDE.md (comprehensive guide)
- QUICKSTART.md (quick start guide)
        """)
    
    except KeyboardInterrupt:
        logger.info("\n\nExamples interrupted by user.")
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
