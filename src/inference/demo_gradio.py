import sys
import os
import tempfile

# ----------------------------
# Fix imports for your folder
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import gradio as gr
from PIL import Image

# Your pipeline
from preprocessing.pipeline import process_note


def ocr_image(img):
    """
    Runs the full OCR + subject classifier pipeline.
    """
    try:
        # Save PIL image to temp file so OpenCV can read it
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            img_path = tmp.name

        # Run complete processing
        result = process_note(img_path)

        # Format output
        if isinstance(result, dict):
            text = result.get("text", "")
            subject = result.get("subject", "Unknown")
            keywords = result.get("keywords_used", [])
            conf = result.get("confidence", None)

            output = (
                f"📄 **Extracted Text:**\n{text}\n\n"
                f"📘 **Subject:** {subject}\n"
                f"🔑 **Keywords:** {keywords}\n"
            )

            if conf is not None:
                output += f"📊 **Confidence:** {conf:.2f}\n"

            return output

        else:
            return f"[Unexpected pipeline output]\nType = {type(result)}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ----------------------------
# Gradio UI
# ----------------------------
demo = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(lines=30, label="Output"),   # ⬅⬅ ONLY CHANGE
    title="SMART-NOTES OCR Demo",
    description="Upload handwritten notes → OCR → Subject classification."
)

if __name__ == "__main__":
    demo.launch()
