"""
demo_gradio_notes.py - FIXED VERSION

Now uses the unified OCR engine that supports both:
- Pytesseract (for printed text)
- CRNN (for handwriting)
"""

import gradio as gr
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Initialize variables BEFORE try block to prevent unbound errors
UnifiedOCREngine = None
recognize_image = None
USE_UNIFIED = False

try:
    from src.model.unified_ocr import UnifiedOCREngine
    USE_UNIFIED = True
    print("[INFO] Using unified OCR engine (Tesseract + CRNN)")
except ImportError:
    # Fallback to old method
    try:
        from preprocessing.recognize import recognize_image
        USE_UNIFIED = False
        print("[WARNING] Using legacy OCR (CRNN only)")
    except ImportError:
        print("[ERROR] No OCR engine available!")
        recognize_image = None

from preprocessing.subject_classifier import classify_subject

# Global OCR engine
ocr_engine = None
if USE_UNIFIED and UnifiedOCREngine is not None:
    try:
        ocr_engine = UnifiedOCREngine(
            checkpoint_path="checkpoints/ocr_epoch_6.pth",
            prefer_tesseract=True,
            confidence_threshold=30
        )
    except Exception as e:
        print(f"[WARNING] Failed to initialize UnifiedOCREngine: {e}")
        ocr_engine = None


def process_note_image(image):
    """
    Process uploaded image and return results
    
    Args:
        image: PIL Image or numpy array from Gradio
        
    Returns:
        tuple: (extracted_text, subject, confidence, keywords_text)
    """
    if image is None:
        return (
            "⚠️ No image uploaded",
            "Unknown Subject",
            "0.00",
            "No keywords detected"
        )
    
    try:
        # Save temporary image
        import tempfile
        import numpy as np
        from PIL import Image
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            temp_img = Image.fromarray(image)
        else:
            temp_img = image
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_img.save(tmp.name)
            temp_path = tmp.name
        
        print(f"[OCR] Processing: {temp_path}")
        
        # Run OCR
        text = ""
        ocr_conf = 0
        engine_used = "none"
        
        if USE_UNIFIED and ocr_engine:
            result = ocr_engine.recognize(temp_path, mode='auto', debug=True)
            text = result['text']
            ocr_conf = result['confidence']
            engine_used = result['engine']
            
            print(f"[OCR] Engine: {engine_used}, Confidence: {ocr_conf:.1f}")
        elif recognize_image is not None:
            text = recognize_image(temp_path)
            ocr_conf = 0
            engine_used = 'crnn'
        else:
            # No OCR engine available
            Path(temp_path).unlink(missing_ok=True)
            return (
                "❌ No OCR engine available\n\n"
                "Please install dependencies:\n"
                "- pip install pytesseract\n"
                "- brew install tesseract  # macOS\n"
                "OR\n"
                "- Ensure CRNN model is available",
                "Error",
                "0.00",
                "No OCR engine"
            )
        
        # Cleanup temp file
        Path(temp_path).unlink(missing_ok=True)
        
        # Check if OCR succeeded
        if not text or len(text.strip()) < 10:
            return (
                "❌ No text detected\n\nTroubleshooting:\n"
                "- Ensure image contains clear text\n"
                "- Try a higher resolution image\n"
                "- Check if text is readable by eye",
                "Unknown Subject",
                "0.00",
                "No text found"
            )
        
        print(f"[CLEANUP] Cleaning OCR output...")
        
        # Clean text
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        print(f"[CLASSIFY] Classifying text...")
        
        # Classify subject
        subject_name, keywords, confidence = classify_subject(text)
        
        # Format output
        confidence_text = f"{confidence:.2f}"
        
        # Format keywords
        if keywords:
            keywords_text = ", ".join(keywords[:10])
            if len(keywords) > 10:
                keywords_text += f" (+ {len(keywords) - 10} more)"
        else:
            keywords_text = "No keywords matched"
        
        # Add metadata to text output
        if USE_UNIFIED:
            text_with_meta = f"[OCR Engine: {engine_used.upper()}, Confidence: {ocr_conf:.1f}%]\n\n{text}"
        else:
            text_with_meta = text
        
        print(f"[OK] Classification: {subject_name} (confidence: {confidence})")
        
        return (
            text_with_meta,
            subject_name,
            confidence_text,
            keywords_text
        )
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error processing image:\n{str(e)}\n\n{traceback.format_exc()}"
        print(f"[ERROR] {e}")
        traceback.print_exc()
        
        return (
            error_msg,
            "Error",
            "0.00",
            "Processing failed"
        )


def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-box {
        min-height: 300px;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, title="SmartNotes OCR") as demo:
        gr.Markdown(
            """
            # 📝 SmartNotes OCR + Subject Classifier
            
            Upload handwritten or printed notes to extract text and automatically classify the subject.
            
            **Features:**
            - 🖨️ **Printed Text**: High accuracy using Tesseract OCR
            - ✍️ **Handwritten Text**: CRNN-based recognition  
            - 📚 **Auto-Classification**: Detects VTU 5th semester subjects
            - 🎯 **Confidence Scoring**: Shows classification reliability
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Image")
                image_input = gr.Image(
                    label="Upload Note Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )
                
                process_btn = gr.Button(
                    "🔍 Extract Text & Classify",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    **Tips:**
                    - Use clear, well-lit images
                    - Ensure text is in focus
                    - Higher resolution = better accuracy
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                
                text_output = gr.Textbox(
                    label="Extracted Text",
                    lines=12,
                    max_lines=20,
                    placeholder="Extracted text will appear here...",
                    show_copy_button=True
                )
                
                subject_output = gr.Textbox(
                    label="Predicted Subject",
                    placeholder="Subject classification..."
                )
                
                confidence_output = gr.Textbox(
                    label="Classification Confidence",
                    placeholder="0.00"
                )
                
                keywords_output = gr.Textbox(
                    label="Matched Keywords",
                    lines=3,
                    placeholder="Keywords will appear here..."
                )
        
        # Examples section
        gr.Markdown("---")
        gr.Markdown("### 📖 System Information")
        
        with gr.Accordion("OCR Engines", open=False):
            if USE_UNIFIED:
                has_tesseract = ocr_engine and ocr_engine.has_tesseract if ocr_engine else False
                has_crnn = ocr_engine and ocr_engine.has_crnn if ocr_engine else False
                
                engine_status = f"""
                **Status:** ✅ Unified OCR Engine Active
                
                **Available Engines:**
                - Pytesseract: {'✅ Available' if has_tesseract else '❌ Not installed'}
                - CRNN: {'✅ Loaded' if has_crnn else '❌ Not loaded'}
                
                **Mode:** Automatic (tries Tesseract first, falls back to CRNN)
                """
            else:
                engine_status = """
                **Status:** ⚠️ Legacy Mode (CRNN only)
                
                To enable Pytesseract support:
                1. Install: `pip install pytesseract`
                2. Install Tesseract: `brew install tesseract` (macOS)
                3. Use the unified_ocr.py module
                """
            
            gr.Markdown(engine_status)
        
        with gr.Accordion("Supported Subjects", open=False):
            gr.Markdown(
                """
                - BCS501 - Software Engineering & Project Management
                - BCS502 - Computer Networks
                - BCS503 - Theory of Computation
                - BCSL504 - Web Technology Lab
                - BCS515A-D - Professional Electives
                - BCS586 - Mini Project
                - BRMK557 - Research Methodology and IPR
                - BCS508 - Environmental Studies
                - General: Biology, Chemistry, Physics, Mathematics
                """
            )
        
        # Connect button to processing function
        process_btn.click(
            fn=process_note_image,
            inputs=image_input,
            outputs=[text_output, subject_output, confidence_output, keywords_output]
        )
    
    return demo


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SmartNotes Gradio Interface")
    print("="*70)
    print("\nStarting server...")
    print("Access at: http://localhost:7860\n")
    
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )