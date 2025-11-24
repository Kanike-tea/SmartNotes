"""
Gradio interface for SmartNotes OCR with Subject Classification

Upload handwritten or printed notes and:
1. Extract text using OCR
2. Classify the subject automatically
3. Display results with confidence scores

UPDATED VERSION with:
- Enhanced CLAHE preprocessing
- Multi-page context classification
- Biology-specific improvements
- Fuzzy keyword matching
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
from collections import defaultdict

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
    Clean and improve garbled OCR output with biology-specific corrections
    """
    if not text:
        return text
    
    import re
    
    # Remove excessive single characters and noise
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Keep words that are at least 2 characters or numbers
        if len(word) >= 2 or word.isdigit():
            cleaned_words.append(word)
        # Also keep single letters if common
        elif word.lower() in ['a', 'i', 'c', 'h', 'o', 'e']:
            cleaned_words.append(word)
    
    # Rejoin cleaned words
    cleaned_text = " ".join(cleaned_words)
    
    # Common OCR error corrections
    corrections = {
        r'\bwl\b': 'will',
        r'\bsel\b': 'cell',
        r'\bcel\b': 'cell',
        r'\bproteln\b': 'protein',
        r'\borgantsm\b': 'organism',
        r'\bmlto\b': 'mito',
        r'\bnucteus\b': 'nucleus',
        r'\bcytopfasm\b': 'cytoplasm',
        r'\bmembrane\b': 'membrane',
    }
    
    for pattern, replacement in corrections.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    # Biology-specific fragment matching
    bio_word_fragments = {
        'cel': 'cell',
        'nuc': 'nucleus',
        'mito': 'mitochondria',
        'chloro': 'chloroplast',
        'prot': 'protein',
        'org': 'organism',
        'tiss': 'tissue',
        'meta': 'metabolism',
    }
    
    # Apply fragment matching
    words_corrected = []
    for word in cleaned_text.split():
        best_match = word
        for fragment, full_word in bio_word_fragments.items():
            if fragment in word.lower() and len(word) >= 3:
                best_match = full_word
                break
        words_corrected.append(best_match)
    
    return " ".join(words_corrected)


def extract_keywords_from_text(text):
    """
    Extract meaningful keywords with fuzzy matching and weighted scoring
    """
    import re
    
    text_lower = text.lower()
    keywords = []
    
    # Enhanced subject patterns with 3-tier weighting
    subject_patterns = {
        'biology': {
            'strong': ['cell', 'dna', 'protein', 'organism', 'tissue', 'enzyme', 
                      'mitochondria', 'nucleus', 'chromosome', 'gene'],
            'medium': ['biology', 'genetic', 'mutation', 'evolution', 'metabolism',
                      'photosynthesis', 'respiration'],
            'weak': ['organ', 'system', 'structure', 'function']
        },
        'chemistry': {
            'strong': ['atom', 'molecule', 'chemical', 'reaction', 'compound'],
            'medium': ['element', 'bonding', 'acid', 'base'],
            'weak': ['solution', 'mixture']
        },
        'physics': {
            'strong': ['force', 'energy', 'motion', 'wave', 'quantum', 'particle'],
            'medium': ['physics', 'velocity', 'acceleration'],
            'weak': ['speed', 'distance']
        },
        'mathematics': {
            'strong': ['equation', 'calculus', 'algebra', 'derivative', 'integral'],
            'medium': ['function', 'theorem', 'proof'],
            'weak': ['number', 'calculate']
        },
        'engineering': {
            'strong': ['circuit', 'design', 'system', 'mechanical', 'electrical'],
            'medium': ['engineering', 'structure'],
            'weak': ['build', 'construct']
        },
    }
    
    # Fuzzy match with weights
    for subject, levels in subject_patterns.items():
        for level, pattern_list in levels.items():
            weight = {'strong': 1.0, 'medium': 0.6, 'weak': 0.3}[level]
            
            for pattern in pattern_list:
                # Exact match
                if pattern in text_lower:
                    keywords.append((f"{subject}:{pattern}", weight))
                # Fuzzy match for longer words
                elif len(pattern) >= 5:
                    for word in text_lower.split():
                        if len(word) >= 4:
                            # Character overlap ratio
                            common_chars = set(pattern) & set(word)
                            if len(common_chars) >= len(pattern) * 0.6:
                                keywords.append((f"{subject}:{pattern}*", weight * 0.7))
    
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
        Convert PDF pages to images and process with MULTI-PAGE CONTEXT
        
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
            page_predictions = []  # NEW: Track per-page classifications
            
            for idx, image in enumerate(images):
                print(f"[PDF] Processing page {idx + 1}/{len(images)}")
                
                # Save image temporarily
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name)
                    image_path = tmp.name
                
                try:
                    # OCR on this page with enhanced preprocessing
                    img_array = np.array(image)
                    
                    # Preprocess for better OCR
                    if len(img_array.shape) == 3:
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_array
                    
                    # ENHANCED PREPROCESSING PIPELINE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray)
                    denoised = cv2.fastNlMeansDenoising(enhanced, h=15, templateWindowSize=7, searchWindowSize=21)
                    binary = cv2.adaptiveThreshold(
                        denoised, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # Save preprocessed image
                    processed_img = Image.fromarray(binary)
                    processed_img.save(image_path)
                    
                    # Run OCR
                    text = self.recognizer.predict(image_path)
                    
                    # Clean text
                    cleaned_text = clean_ocr_text(text)
                    
                    if cleaned_text and cleaned_text != "[NO TEXT DETECTED]":
                        all_text.append(f"--- Page {idx + 1} ---\n{cleaned_text}")
                        
                        # NEW: Get per-page prediction for voting
                        page_subject, page_keywords, page_conf = classify_subject(cleaned_text)
                        page_predictions.append((page_subject, page_conf, page_keywords))
                finally:
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except:
                            pass
            
            # Combine all text
            combined_text = "\n".join(all_text)
            
            if not combined_text:
                return "", "No text detected in PDF", 0.0, [], len(images)
            
            # MULTI-PAGE CONTEXT: Vote across all pages
            if len(page_predictions) > 1:
                print(f"[MULTI-PAGE] Using document-level context ({len(page_predictions)} pages)")
                
                # Aggregate confidence scores by subject
                subject_scores = defaultdict(float)
                all_keywords = set()
                
                for subj, conf, kws in page_predictions:
                    subject_scores[subj] += conf
                    all_keywords.update(kws)
                
                # Best subject = highest total confidence across pages
                best_subject = max(subject_scores, key=lambda k: subject_scores[k]) if subject_scores else "Unknown Subject"
                avg_confidence = subject_scores[best_subject] / len(page_predictions) if best_subject in subject_scores else 0.0
                
                # Use document-level classification
                subject = best_subject
                confidence = min(avg_confidence, 1.0)
                keywords = list(all_keywords)[:10]
                
                print(f"[DOCUMENT] Classification: {subject} (avg conf: {confidence:.2f})")
            else:
                # Single page - use standard classification
                subject, keywords, confidence = classify_subject(combined_text)
            
            # If still low confidence, try enhanced matching
            if confidence < 0.3:
                extra_keywords = extract_keywords_from_text(combined_text)
                if extra_keywords:
                    subject_scores = {}
                    for kw_match, weight in extra_keywords:
                        subj = kw_match.split(':')[0]
                        if subj not in subject_scores:
                            subject_scores[subj] = 0
                        subject_scores[subj] += weight
                    
                    if subject_scores:
                        best_subject_key = max(subject_scores, key=lambda k: subject_scores[k]) if subject_scores else None
                        if best_subject_key:
                            enhanced_conf = min(subject_scores[best_subject_key] / 5.0, 1.0)
                        else:
                            enhanced_conf = 0.0
                        
                        if enhanced_conf > confidence and best_subject_key:
                            subject_map = {
                                'biology': 'Biology',
                                'chemistry': 'Chemistry',
                                'physics': 'Physics',
                                'mathematics': 'Mathematics',
                                'engineering': 'Engineering'
                            }
                            subject = subject_map.get(best_subject_key, subject)
                            confidence = enhanced_conf
                            keywords.extend([kw.split(':')[1].rstrip('*') for kw, _ in extra_keywords[:5]])
            
            return combined_text, subject, confidence, keywords, len(images)
        
        except Exception as e:
            print(f"[ERROR] PDF processing failed: {e}")
            return f"Error: {str(e)}", "Error", 0.0, [], 0
    
    def process_image(self, image_input):
        """
        Process uploaded image with ENHANCED PIPELINE:
        1. Advanced preprocessing (CLAHE + denoising + adaptive threshold)
        2. Extract text via OCR
        3. Clean and improve extracted text
        4. Classify subject with multi-level fallback
        5. Return results
        
        Args:
            image_input: PIL Image or file path
            
        Returns:
            Tuple of (extracted_text, subject, confidence, keywords)
        """
        try:
            # Convert PIL Image to file path
            if isinstance(image_input, Image.Image):
                # ENHANCED PREPROCESSING PIPELINE
                img_array = np.array(image_input)
                
                # Convert to grayscale if needed
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                # Step 1: CLAHE for contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Step 2: Aggressive denoising
                denoised = cv2.fastNlMeansDenoising(enhanced, h=15, templateWindowSize=7, searchWindowSize=21)
                
                # Step 3: Adaptive thresholding
                binary = cv2.adaptiveThreshold(
                    denoised, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Step 4: Morphological operations
                kernel = np.ones((2, 2), np.uint8)
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Convert back to PIL and save
                processed_img = Image.fromarray(cleaned)
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
                print(f"[WARNING] Poor OCR quality detected")
                return extracted_text, "Unable to extract clear text - try clearer image", 0.0, []
            
            # Step 3: Subject Classification with enhanced keyword extraction
            print(f"[CLASSIFY] Classifying text...")
            subject, keywords, confidence = classify_subject(cleaned_text)
            
            # If confidence is very low, try enhanced keyword extraction
            if confidence < 0.3:
                print(f"[LOW_CONFIDENCE] Confidence {confidence} - using enhanced keyword matching...")
                extra_keywords = extract_keywords_from_text(cleaned_text)
                
                if extra_keywords:
                    print(f"[FOUND] Additional context: {len(extra_keywords)} fuzzy matches")
                    
                    # Aggregate scores by subject
                    subject_scores = {}
                    for kw_match, weight in extra_keywords:
                        subj = kw_match.split(':')[0]
                        if subj not in subject_scores:
                            subject_scores[subj] = 0
                        subject_scores[subj] += weight
                    
                    if subject_scores:
                        # Re-classify based on aggregated scores
                        best_subject_key = max(subject_scores, key=lambda k: subject_scores[k]) if subject_scores else None
                        if best_subject_key:
                            enhanced_conf = min(subject_scores[best_subject_key] / 5.0, 1.0)
                        else:
                            enhanced_conf = 0.0
                        
                        if enhanced_conf > confidence and best_subject_key:
                            # Map to full subject name
                            subject_map = {
                                'biology': 'Biology',
                                'chemistry': 'Chemistry',
                                'physics': 'Physics',
                                'mathematics': 'Mathematics',
                                'engineering': 'Engineering'
                            }
                            subject = subject_map.get(best_subject_key, subject)
                            confidence = enhanced_conf
                            keywords.extend([kw.split(':')[1].rstrip('*') for kw, _ in extra_keywords[:5]])
                            print(f"[ENHANCED] New classification: {subject} (confidence: {confidence:.2f})")
            
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
        
        ✨ **NEW**: Enhanced preprocessing with CLAHE + Multi-page context classification
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