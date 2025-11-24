"""
Preprocessing module for SmartNotes.

Includes:
  - Line segmentation from raw images
  - Text recognition pipeline
  - Post-processing and validation
  - Subject classification
"""

from .pipeline import process_note
from .recognize import recognize_image, OCRRecognizer
from .subject_classifier import classify_subject

__all__ = [
    "process_note",
    "recognize_image",
    "OCRRecognizer",
    "classify_subject",
]
