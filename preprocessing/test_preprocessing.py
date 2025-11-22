"""
Simple pytest test for the preprocessing pipeline.

This file previously attempted to parse command-line arguments at import time,
which caused pytest to error during collection. Convert this into a pytest
compatible test that creates a temporary image and runs the pipeline.
"""
import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

# Ensure package root is on sys.path (for local imports)
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocessing.pipeline import process_note


def test_process_note_runs_on_blank_image():
    """Test that process_note returns a dict on a valid image path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test_img.png")

        # Create a blank white image (32x128), compatible with the recognizer
        img = 255 * (1.0 * np.ones((128, 32), dtype=np.uint8)).T
        # Ensure correct orientation: recognizer expects (H=32, W=128)
        img = img.reshape((32, 128))
        cv2.imwrite(img_path, img)

        # Run pipeline
        output = process_note(img_path)

        assert isinstance(output, dict)
        # Ensure keys are present
        assert 'text' in output
        assert 'subject' in output
        assert 'keywords_used' in output
        assert 'confidence' in output
