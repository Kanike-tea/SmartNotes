# text_preprocess.py — (Anushka Future Work)

import re

def clean_text(text):
    """
    Remove extra spaces, clean symbols, standardize formatting.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text

