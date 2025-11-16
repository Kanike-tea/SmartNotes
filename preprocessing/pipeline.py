# pipeline.py 

from .recognize import recognize_image
from .subject_classifier import classify_subject


def process_note(image_path):
    """
    Complete pipeline:
    1. Run OCR
    2. Run subject classifier
    3. Return clean structured dictionary
    """
    text = recognize_image(image_path)
    subject = classify_subject(text)

    result = {
        "text": text,
        "subject": subject,
        "keywords_used": subject if subject != "Unknown Subject" else []
    }

    return result
