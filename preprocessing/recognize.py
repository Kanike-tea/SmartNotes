# recognize.py — CLEANED, FIXED, STABLE FULL-PAGE OCR VERSION

import sys
import platform
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import cv2
import numpy as np

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer
from preprocessing.line_segment import segment_lines

# Language Model (optional)
HAS_LM = False
try:
    from pyctcdecode import build_ctcdecoder
    HAS_LM = True
    print("[INFO] pyctcdecode detected — LM decoding available.")
except ImportError:
    print("[INFO] pyctcdecode not installed — LM disabled.")


# ======================================================
#   MAIN OCR CLASS
# ======================================================
class OCRRecognizer:
    def __init__(
        self,
        checkpoint_path=None
    ):
        self.device = torch.device("cpu")
        self.tokenizer = TextTokenizer()
        self.num_classes = len(self.tokenizer.chars)

        # -----------------------------
        # Default relative path to checkpoint
        # -----------------------------
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).resolve().parents[1] / "model" / "checkpoints" / "ocr_finetuned_stage2_best.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        # -----------------------------
        # Load CRNN model
        # -----------------------------
        self.model = CRNN(num_classes=self.num_classes).to(self.device)

        if checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"[OK] Loaded OCR model from: {checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint NOT FOUND at:\n{checkpoint_path}")
            print("[WARNING] Using placeholder predictions!")

        self.model.eval()

        # -----------------------------
        # Load LM (optional)
        # -----------------------------
        self.decoder = None
        self.use_lm = False

        if platform.system() == "Windows" and HAS_LM:
            lm_path = Path(__file__).resolve().parents[1] / "lm" / "smartnotes.arpa"

            vocab = list(self.tokenizer.chars) + ["-"]

            if lm_path.exists():
                print("[OK] Loading LM decoder...")
                self.decoder = build_ctcdecoder(
                    labels=vocab,
                    kenlm_model_path=str(lm_path)
                )
                self.use_lm = True
            else:
                print("[INFO] LM not found — falling back to greedy decoding.")

    # --------------------------------------------------
    # Preprocess a single text line
    # --------------------------------------------------
    def preprocess_line(self, img):
        img = cv2.resize(img, (128, 32))
        img = img / 255.0
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    # --------------------------------------------------
    # Predict a single line of text
    # --------------------------------------------------
    def predict_line(self, line_img):
        img = self.preprocess_line(line_img)

        with torch.no_grad():
            preds = self.model(img)
            preds = preds.permute(1, 0, 2).cpu()
            probs = torch.softmax(preds, dim=2).numpy()

        # Language model decoding
        if self.use_lm:
            log_probs = np.log(probs[0] + 1e-9)
            return self.decoder.decode(log_probs).strip()

        # Greedy decoding fallback
        seq = torch.argmax(preds[0], dim=1).numpy()
        return self.tokenizer.decode(seq).strip()

    # --------------------------------------------------
    # Perform full-page OCR (line-by-line)
    # --------------------------------------------------
    def predict(self, image_path):
        lines = segment_lines(image_path)

        print(f"[DEBUG] Detected {len(lines)} lines")

        if len(lines) == 0:
            return "[NO TEXT DETECTED]"

        output_lines = []
        for line in lines:
            text = self.predict_line(line)
            if text:
                output_lines.append(text)

        return "\n".join(output_lines)


# ======================================================
#   STATIC WRAPPER FUNCTION
# ======================================================
# Create a SINGLE recognizer instance (efficiency)
_global_recognizer = None

def recognize_image(image_path):
    global _global_recognizer

    if _global_recognizer is None:
        _global_recognizer = OCRRecognizer()

    return _global_recognizer.predict(image_path)


# ======================================================
#   CLI SUPPORT
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    recognizer = OCRRecognizer(checkpoint_path=args.checkpoint)
    text = recognizer.predict(args.image)

    print("\n===== OCR OUTPUT =====")
    print(text)
    print("======================\n")
