# recognize.py  — FINAL PRODUCTION VERSION

import sys
import platform
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import cv2
import numpy as np

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer

# Try LM support
HAS_LM = False
try:
    from pyctcdecode import build_ctcdecoder
    HAS_LM = True
    print("[INFO] pyctcdecode detected — LM decoding available.")
except ImportError:
    HAS_LM = False
    print("[INFO] pyctcdecode not installed — LM decoding disabled.")


class OCRRecognizer:
    def __init__(self, checkpoint_path="checkpoints/ocr_finetuned_stage2_best.pth"):

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = TextTokenizer()
        self.num_classes = len(self.tokenizer.chars)

        # Load model
        self.model = CRNN(num_classes=self.num_classes).to(self.device)

        # Load weights
        if Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[OK] Loaded OCR model from: {checkpoint_path}")
        else:
            print("[WARNING] Model checkpoint NOT found — using placeholder OCR output.")

        self.model.eval()

        # ---- Optional LM (Windows only) ----
        self.decoder = None
        self.use_lm = False

        if platform.system() == "Windows" and HAS_LM:
            lm_path = Path("lm/smartnotes.arpa")

            vocab = list(self.tokenizer.chars) + ["-"]

            if lm_path.exists():
                print("[OK] Loading LM decoder (Windows mode)...")
                self.decoder = build_ctcdecoder(
                    labels=vocab,
                    kenlm_model_path=str(lm_path)
                )
                self.use_lm = True
            else:
                print("[INFO] No LM (.arpa) file found — beam search disabled.")

        else:
            print("[INFO] LM decoding skipped (Linux/Mac or missing pyctcdecode).")

    # -----------------------------------------------------
    # IMAGE PREPROCESSING
    # -----------------------------------------------------
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = img / 255.0
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    # -----------------------------------------------------
    # PREDICT USING GREEDY OR LM
    # -----------------------------------------------------
    def predict(self, image_path):

        img = self.preprocess_image(image_path)

        with torch.no_grad():
            preds = self.model(img)              # (T, B, C)
            preds = preds.permute(1, 0, 2).cpu() # (B, T, C)
            probs = torch.softmax(preds, dim=2).numpy()

        # ---- LM decoding (Windows) ----
        if self.use_lm:
            log_probs = np.log(probs[0] + 1e-9)
            text = self.decoder.decode(log_probs)
            return text.strip()

        # ---- Greedy decoding ----
        seq = torch.argmax(preds[0], dim=1).numpy()
        text = self.tokenizer.decode(seq)
        return text.strip()


# STATIC FUNCTION
def recognize_image(image_path):
    recognizer = OCRRecognizer()
    return recognizer.predict(image_path)


# CLI ENTRYPOINT
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ocr_finetuned_stage2_best.pth")
    args = parser.parse_args()

    recognizer = OCRRecognizer(checkpoint_path=args.checkpoint)

    text = recognizer.predict(args.image)

    print("\n===== OCR OUTPUT =====")
    print(text)
    print("======================\n")
