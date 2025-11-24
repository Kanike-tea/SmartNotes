#!/usr/bin/env python3
"""
SmartNotes Gradio Launcher
Easy startup script for the Notes OCR + Subject Classification interface
"""

import sys
import os
from pathlib import Path

# FIXED: Set working directory to SmartNotes root
smartnotes_root = Path(__file__).parent.parent.resolve()
os.chdir(smartnotes_root)
sys.path.insert(0, str(smartnotes_root))

print("\n" + "=" * 70)
print("SmartNotes Gradio Interface - Starting")
print("=" * 70)
print(f"Working directory: {smartnotes_root}")
print("\nFeatures:")
print("   ✓ Upload handwritten or printed notes")
print("   ✓ Extract text using OCR (CRNN + BiLSTM)")
print("   ✓ Classify subject automatically")
print("   ✓ Show classification confidence\n")

# Check dependencies
try:
    import gradio
    import torch
    from preprocessing.subject_classifier import classify_subject
    from preprocessing.recognize import OCRRecognizer
    print("✓ All dependencies available\n")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with: pip install -r requirements.txt")
    print("Also install: pip install gradio")
    sys.exit(1)

# Launch interface
print("Starting web interface...")
print("   Access at: http://localhost:7860")
print("   Press CTRL+C to stop\n")

from src.inference.demo_gradio_notes import create_gradio_interface

interface = create_gradio_interface()
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True
)