# recognize.py — CLEAN, FIXED, WORKING VERSION

import sys
import platform
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import cv2
import numpy as np

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer
from preprocessing.line_segment import segment_lines


# ==============================
# Optional LM
# ==============================
HAS_LM = False
try:
    from pyctcdecode import build_ctcdecoder
    HAS_LM = True
    print("[INFO] pyctcdecode detected — LM decoding available.")
except ImportError:
    print("[INFO] pyctcdecode not installed — LM disabled.")


class OCRRecognizer:
    def __init__(self,
                 checkpoint_path="src/model/checkpoints/ocr_finetuned_stage2_best.pth"):

        self.device = torch.device("cpu")
        self.tokenizer = TextTokenizer()
        self.num_classes = len(self.tokenizer.chars)

        # Load CRNN model
        self.model = CRNN(num_classes=self.num_classes).to(self.device)

        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"[OK] Loaded OCR checkpoint: {checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint NOT FOUND at:\n{checkpoint_path}")
            print("[WARNING] OCR output will be incorrect!")

        self.model.eval()

        # Optional LM (Windows only)
        self.decoder = None
        self.use_lm = False

        if platform.system() == "Windows" and HAS_LM:
            lm_path = Path("lm/smartnotes.arpa")

            vocab = list(self.tokenizer.chars) + ["-"]

            if lm_path.exists():
                print("[OK] Loading LM decoder...")
                self.decoder = build_ctcdecoder(
                    labels=vocab,
                    kenlm_model_path=str(lm_path)
                )
                self.use_lm = True
            else:
                print("[INFO] LM file not found — using greedy decoding.")

    # --------------------------------------
    # Preprocess line
    # --------------------------------------
    def preprocess_line(self, img):
        img = cv2.resize(img, (128, 32))
        img = img / 255.0
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    # --------------------------------------
    # Predict single line
    # --------------------------------------
    def predict_line(self, line_img):
        img = self.preprocess_line(line_img)

        with torch.no_grad():
            preds = self.model(img)
            preds = preds.permute(1, 0, 2).cpu()
            probs = torch.softmax(preds, dim=2).numpy()

        if self.use_lm:
            log_probs = np.log(probs[0] + 1e-9)
            return self.decoder.decode(log_probs).strip()

        seq = torch.argmax(preds[0], dim=1).numpy()
        return self.tokenizer.decode(seq).strip()

    # --------------------------------------
    # Full-page OCR
    # --------------------------------------
    def predict(self, image_path):
        lines = segment_lines(image_path)

        print(f"[DEBUG] Detected {len(lines)} lines")

        if len(lines) == 0:
            return "[NO TEXT DETECTED]"

        result = []
        for line in lines:
            text = self.predict_line(line)
            if text:
                result.append(text)

        return "\n".join(result)


# ========================================
# STATIC WRAPPER (used by pipeline)
# ========================================
_global_ocr = None

def recognize_image(image_path):
    global _global_ocr
    if _global_ocr is None:
        _global_ocr = OCRRecognizer()

    return _global_ocr.predict(image_path)


# ========================================
# CLI SUPPORT
# ========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str,
        default="src/model/checkpoints/ocr_finetuned_stage2_best.pth")
    args = parser.parse_args()

    recognizer = OCRRecognizer(checkpoint_path=args.checkpoint)
    output = recognizer.predict(args.image)

    print("\n===== OCR OUTPUT =====")
    print(output)
    print("======================\n")
