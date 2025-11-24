#!/usr/bin/env python3
"""
SmartNotes Gradio Launcher
Easy startup script for the Notes OCR + Subject Classification interface
"""

import sys
import os
from pathlib import Path

# Setup imports using path utilities
from smartnotes.paths import setup_imports, get_project_root

setup_imports()
smartnotes_root = get_project_root()
os.chdir(smartnotes_root)

print("\n" + "=" * 70)
print("SmartNotes Gradio Interface - Starting")
print("=" * 70)
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
