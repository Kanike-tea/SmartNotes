# recognize.py 

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import cv2
import numpy as np

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer


class OCRRecognizer:
    def __init__(self, checkpoint_path="checkpoints/ocr_finetuned_stage2_best.pth"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = TextTokenizer()

        # Load model
        num_classes = len(self.tokenizer.chars)
        self.model = CRNN(num_classes=num_classes).to(self.device)

        # Load weights
        if Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[OK] Loaded OCR model from: {checkpoint_path}")
        else:
            print("[WARNING] Model checkpoint not found! Running placeholder mode.")

        self.model.eval()

    def preprocess_image(self, image_path):
        """Load & preprocess image for the model."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))     # (W, H)
        img = img / 255.0
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return img.to(self.device)

    def predict(self, image_path):
        """Run OCR on an image and return extracted text."""
        # if no model exists → return placeholder
        if not hasattr(self.model, "state_dict"):
            return "[OCR model not loaded]"

        img = self.preprocess_image(image_path)

        with torch.no_grad():
            preds = self.model(img)  # (T, B, C)
            preds = preds.permute(1, 0, 2).cpu()

        seq = torch.argmax(preds[0], dim=1).numpy()
        text = self.tokenizer.decode(seq)

        return text.strip()


# Simple function for easy pipeline import
def recognize_image(image_path):
    recognizer = OCRRecognizer()
    return recognizer.predict(image_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    recognizer = OCRRecognizer()
    text = recognizer.predict(args.image)

    print("\n===== OCR OUTPUT =====")
    print(text)
    print("======================\n")

