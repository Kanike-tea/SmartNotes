import sys
import os
import tempfile

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import gradio as gr
from PIL import Image

# Import preprocessing function
from preprocessing.pipeline import process_note

def ocr_image(img):
    """
    Runs preprocessing on the uploaded image and returns a placeholder OCR output.
    Handles OpenCV file path requirement and dictionary output.
    """
    try:
        # Save PIL image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            img_path = tmp.name

        # Run preprocessing
        processed = process_note(img_path)

        # If processed is a dict, show keys; otherwise show type
        if isinstance(processed, dict):
            processed_info = f"Processed keys: {list(processed.keys())}"
        else:
            processed_info = f"Processed type: {type(processed)}"

        # Return info + placeholder OCR
        return f"Image processed successfully!\n{processed_info}\nOCR output placeholder — replace with real model later"

    except Exception as e:
        return f"Error during preprocessing: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="SMART-OCR Demo (Preprocessing Enabled)"
)

demo.launch()
