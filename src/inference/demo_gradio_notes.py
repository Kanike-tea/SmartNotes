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

convert_from_path = None
try:
    from pdf2image import convert_from_path as _convert_from_path
    convert_from_path = _convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARNING] pdf2image not installed. PDF support disabled.")

from preprocessing.recognize import OCRRecognizer
from preprocessing.subject_classifier import classify_subject
from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer


def clean_ocr_text(text):
    """
    Clean and improve garbled OCR output
    
    Args:
        text: Raw OCR extracted text
        
    Returns:
        Cleaned text with improved readability
    """
    if not text:
        return text
    
    import re
    
    # Remove excessive single characters and noise
    # Keep words but clean up garbage
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Keep words that are at least 2 characters
        # Or numbers
        if len(word) >= 2 or word.isdigit():
            cleaned_words.append(word)
        # Also keep single letters if they appear in context (like 'a', 'I', 'c')
        elif word.lower() in ['a', 'i', 'c', 'h', 'o', 'e']:
            cleaned_words.append(word)
    
    # Rejoin cleaned words
    cleaned_text = " ".join(cleaned_words)
    
    # Try to improve readability by common OCR error corrections
    # These are common misreadings in handwritten text
    corrections = {
        r'\bwl\b': 'will',
        r'\bsel\b': 'cell',
        r'\bhal\b': 'hal',
        r'\bl\b': 'l',
        r'\be\b': 'e',
        r'\bh\b': 'h',
        r'\bc\b': 'c',
        r'\bfrom\b': 'from',
        r'\bthe\b': 'the',
        r'\band\b': 'and',
    }
    
    for pattern, replacement in corrections.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text


def extract_keywords_from_text(text):
    """
    Extract meaningful keywords from OCR text for better classification
    when OCR output is poor
    
    Args:
        text: Input text
        
    Returns:
        List of potential keywords
    """
    import re
    
    # Convert to lowercase and split
    text_lower = text.lower()
    
    # Extract multi-word phrases and key terms
    keywords = []
    
    # Common subject keywords patterns
    subject_patterns = {
        'biology': ['cell', 'dna', 'protein', 'organism', 'biology', 'genetic', 'mutation', 'evolution'],
        'chemistry': ['atom', 'molecule', 'chemical', 'reaction', 'element', 'compound', 'bonding'],
        'physics': ['force', 'energy', 'motion', 'wave', 'quantum', 'particle', 'physics'],
        'mathematics': ['equation', 'calculus', 'algebra', 'geometry', 'derivative', 'integral'],
        'engineering': ['circuit', 'design', 'system', 'engineering', 'mechanical', 'electrical'],
    }
    
    # Search for pattern keywords
    for subject, pattern_list in subject_patterns.items():
        for pattern in pattern_list:
            if pattern in text_lower:
                keywords.append(f"{subject}:{pattern}")
    
    return keywords


class NotesProcessor:
    """Process uploaded notes: OCR + Subject Classification"""
    
    def __init__(self):
        """Initialize OCR recognizer and tokenizer"""
        self.recognizer = OCRRecognizer()
        self.tokenizer = TextTokenizer()
        print("✓ NotesProcessor initialized")
    
    def process_pdf(self, pdf_path):
        """
        Convert PDF pages to images and process each with improved OCR
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (all_text, subject, confidence, keywords, page_count)
        """
        if not PDF_SUPPORT:
            return "", "PDF support not available", 0.0, [], 0
        
        try:
            print(f"[PDF] Converting PDF: {pdf_path}")
            if convert_from_path is None:
                raise RuntimeError("pdf2image.convert_from_path is not available despite PDF_SUPPORT=True")
            images = convert_from_path(pdf_path)
            print(f"[PDF] Extracted {len(images)} pages")
            
            all_text = []
            
            for idx, image in enumerate(images):
                print(f"[PDF] Processing page {idx + 1}/{len(images)}")
                
                # Save image temporarily
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name)
                    image_path = tmp.name
                
                try:
                    # OCR on this page with preprocessing
                    img_array = np.array(image)
                    
                    # Preprocess for better OCR
                    if len(img_array.shape) == 3:
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_array
                    
                    # Enhance contrast and denoise
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray)
                    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
                    
                    # Save preprocessed image
                    processed_img = Image.fromarray(denoised)
                    processed_img.save(image_path)
                    
                    # Run OCR
                    text = self.recognizer.predict(image_path)
                    
                    # Clean text
                    cleaned_text = clean_ocr_text(text)
                    
                    if cleaned_text and cleaned_text != "[NO TEXT DETECTED]":
                        all_text.append(f"--- Page {idx + 1} ---\n{cleaned_text}")
                finally:
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except:
                            pass
            
            # Combine all text and classify
            combined_text = "\n".join(all_text)
            
            if not combined_text:
                return "", "No text detected in PDF", 0.0, [], len(images)
            
            subject, keywords, confidence = classify_subject(combined_text)
            
            # If low confidence, add context keywords
            if confidence < 0.3:
                extra_keywords = extract_keywords_from_text(combined_text)
                keywords.extend(extra_keywords)
            
            return combined_text, subject, confidence, keywords, len(images)
        
        except Exception as e:
            print(f"[ERROR] PDF processing failed: {e}")
            return f"Error: {str(e)}", "Error", 0.0, [], 0
    
    def process_image(self, image_input):
        """
        Process uploaded image:
        1. Preprocess image for better OCR
        2. Extract text via OCR
        3. Clean and improve extracted text
        4. Classify subject
        5. Return results
        
        Args:
            image_input: PIL Image or file path
            
        Returns:
            Tuple of (extracted_text, subject, confidence, keywords)
        """
        try:
            # Convert PIL Image to file path
            if isinstance(image_input, Image.Image):
                # Preprocess image for better OCR accuracy
                img_array = np.array(image_input)
                
                # Convert to grayscale if needed
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                # Apply image preprocessing
                # Increase contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Denoise
                denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
                
                # Convert back to PIL and save
                processed_img = Image.fromarray(denoised)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    processed_img.save(tmp.name)
                    image_path = tmp.name
            else:
                image_path = image_input.name if hasattr(image_input, 'name') else str(image_input)
            
            # Step 1: OCR
            print(f"[OCR] Processing: {image_path}")
            extracted_text = self.recognizer.predict(image_path)
            
            # Step 2: Clean extracted text
            print(f"[CLEANUP] Cleaning OCR output...")
            cleaned_text = clean_ocr_text(extracted_text)
            
            if not cleaned_text or cleaned_text == "[NO TEXT DETECTED]":
                # If OCR completely fails, try to extract any meaningful content
                print(f"[WARNING] Poor OCR quality detected")
                return extracted_text, "Unable to extract clear text - try clearer image", 0.0, []
            
            # Step 3: Subject Classification
            print(f"[CLASSIFY] Classifying text...")
            subject, keywords, confidence = classify_subject(cleaned_text)
            
            # If confidence is very low, try to extract additional context
            if confidence < 0.3:
                print(f"[LOW_CONFIDENCE] Confidence {confidence} - searching for context clues...")
                # Extract any additional keywords that might help
                extra_keywords = extract_keywords_from_text(cleaned_text)
                if extra_keywords:
                    print(f"[FOUND] Additional context: {extra_keywords}")
                    keywords.extend(extra_keywords)
            
            # Cleanup temp file
            if isinstance(image_input, Image.Image) and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            return cleaned_text, subject, confidence, keywords
        
        except Exception as e:
            print(f"[ERROR] {e}")
            return f"Error: {str(e)}", "Error", 0.0, []
    
    def process_notes(self, image_input=None, pdf_input=None):
        """
        Process either image or PDF
        
        Args:
            image_input: PIL Image or file path
            pdf_input: PDF file path
            
        Returns:
            Tuple of (extracted_text, subject, confidence, keywords_str, info_str)
        """
        if pdf_input is not None:
            text, subject, confidence, keywords, page_count = self.process_pdf(pdf_input)
            info = f"PDF ({page_count} pages)"
            return text, subject, confidence, keywords, info
        elif image_input is not None:
            text, subject, confidence, keywords = self.process_image(image_input)
            info = "Image"
            return text, subject, confidence, keywords, info
        else:
            return "", "No input provided", 0.0, [], ""


def create_gradio_interface():
    """Create and launch Gradio interface"""
    
    processor = NotesProcessor()
    
    def process(image, pdf_file):
        """Process image or PDF and return results"""
        if image is None and pdf_file is None:
            return "", "No input uploaded", 0.0, "", ""
        
        text, subject, confidence, keywords, info = processor.process_notes(
            image_input=image, 
            pdf_input=pdf_file
        )
        keywords_str = ", ".join(keywords) if keywords else "None"
        
        return text, subject, confidence, keywords_str, info
    
    # Create interface
    interface = gr.Interface(
        fn=process,
        inputs=[
            gr.Image(label="Upload Image (Handwritten or Printed)", type="pil"),
            gr.File(label="Upload PDF", file_types=[".pdf"])
        ],
        outputs=[
            gr.Textbox(label="Extracted Text", lines=10),
            gr.Textbox(label="Predicted Subject"),
            gr.Number(label="Classification Confidence", precision=3),
            gr.Textbox(label="Matched Keywords"),
            gr.Textbox(label="Source Info")
        ],
        title="📖 SmartNotes - OCR & Subject Classification",
        description="""
        Upload images or PDFs of handwritten or printed course notes.
        
        The system will:
        1. **Extract Text** using deep learning OCR (CRNN + BiLSTM)
        2. **Classify Subject** based on extracted keywords
        3. **Show Confidence** score for the classification
        
        Supports: Handwritten notes, printed images, PDF documents
        """,
        examples=[
            # You can add example images here if available
        ],
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
