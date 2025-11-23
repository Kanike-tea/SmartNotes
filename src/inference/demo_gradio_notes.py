"""
Gradio interface for SmartNotes OCR with Subject Classification

Upload handwritten or printed notes and:
1. Extract text using OCR
2. Classify the subject automatically
3. Display results with confidence scores
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile

from preprocessing.recognize import OCRRecognizer
from preprocessing.subject_classifier import classify_subject
from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer


class NotesProcessor:
    """Process uploaded notes: OCR + Subject Classification"""
    
    def __init__(self):
        """Initialize OCR recognizer and tokenizer"""
        self.recognizer = OCRRecognizer()
        self.tokenizer = TextTokenizer()
        print("✓ NotesProcessor initialized")
    
    def process_notes(self, image_input):
        """
        Process uploaded image:
        1. Extract text via OCR
        2. Classify subject
        3. Return results
        
        Args:
            image_input: PIL Image or file path
            
        Returns:
            Tuple of (extracted_text, subject, confidence, keywords)
        """
        try:
            # Convert PIL Image to file path
            if isinstance(image_input, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image_input.save(tmp.name)
                    image_path = tmp.name
            else:
                image_path = image_input.name if hasattr(image_input, 'name') else str(image_input)
            
            # Step 1: OCR
            print(f"[OCR] Processing: {image_path}")
            extracted_text = self.recognizer.predict(image_path)
            
            if not extracted_text or extracted_text == "[NO TEXT DETECTED]":
                return "", "Unable to detect text", 0.0, []
            
            # Step 2: Subject Classification
            print(f"[CLASSIFY] Classifying text...")
            subject, keywords, confidence = classify_subject(extracted_text)
            
            # Cleanup temp file
            if isinstance(image_input, Image.Image) and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            return extracted_text, subject, confidence, keywords
        
        except Exception as e:
            print(f"[ERROR] {e}")
            return f"Error: {str(e)}", "Error", 0.0, []


def create_gradio_interface():
    """Create and launch Gradio interface"""
    
    processor = NotesProcessor()
    
    def process(image):
        """Process image and return results"""
        if image is None:
            return "", "No image uploaded", 0.0, ""
        
        text, subject, confidence, keywords = processor.process_notes(image)
        keywords_str = ", ".join(keywords) if keywords else "None"
        
        return text, subject, confidence, keywords_str
    
    # Create interface
    interface = gr.Interface(
        fn=process,
        inputs=gr.Image(label="Upload Notes (Handwritten or Printed)", type="pil"),
        outputs=[
            gr.Textbox(label="Extracted Text", lines=8),
            gr.Textbox(label="Predicted Subject"),
            gr.Number(label="Classification Confidence", precision=3),
            gr.Textbox(label="Matched Keywords")
        ],
        title="📖 SmartNotes - OCR & Subject Classification",
        description="""
        Upload images of handwritten or printed course notes.
        
        The system will:
        1. **Extract Text** using deep learning OCR (CRNN + BiLSTM)
        2. **Classify Subject** based on extracted keywords
        3. **Show Confidence** score for the classification
        
        Supports: Handwritten notes, printed PDFs, photographs
        """,
        examples=[
            # You can add example images here if available
        ],
        # Do not specify a theme to remain compatible with gradio versions
        # that do not export `gr.themes`; the default theme will be used.
        allow_flagging="never"
    )
    
    return interface


if __name__ == "__main__":
    print("=" * 70)
    print("SmartNotes Gradio Interface - OCR + Subject Classification")
    print("=" * 70)
    
    interface = create_gradio_interface()
    
    # Launch with public link
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True
    )
