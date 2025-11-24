# recognize.py — FINAL FULL-PAGE OCR VERSION

import sys
import platform
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import cv2
import numpy as np

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer
from preprocessing.line_segment import segment_lines

# Try LM support
HAS_LM = False
build_ctcdecoder = None  # type: ignore
try:
    from pyctcdecode.decoder import build_ctcdecoder
    HAS_LM = True
    print("[INFO] pyctcdecode detected — LM decoding available.")
except (ImportError, AttributeError):
    print("[INFO] pyctcdecode not installed — LM decoding disabled.")


class OCRRecognizer:
    def __init__(self, checkpoint_path="checkpoints/ocr_finetuned_stage2_best.pth"):

        self.device = torch.device("cpu")
        self.tokenizer = TextTokenizer()
        self.num_classes = len(self.tokenizer.chars)

        # Load CRNN model
        self.model = CRNN(num_classes=self.num_classes).to(self.device)

        if Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[OK] Loaded OCR model from: {checkpoint_path}")
        else:
            print("[WARNING] No checkpoint found — using placeholder predictions.")

        self.model.eval()

        # Optional LM
        self.decoder = None
        self.use_lm = False

        if platform.system() == "Windows" and HAS_LM and build_ctcdecoder is not None:
            lm_path = Path("lm/smartnotes.arpa")
            vocab = list(self.tokenizer.chars) + ["-"]

            if lm_path.exists():
                print("[OK] Loading LM decoder...")
                self.decoder = build_ctcdecoder(  # type: ignore
                    labels=vocab,
                    kenlm_model_path=str(lm_path)
                )
                self.use_lm = True
            else:
                print("[INFO] LM not found — using greedy decoding.")

    # --------------------------
    # Preprocess single line
    # --------------------------
    def preprocess_line(self, img):
        """Enhanced preprocessing with CLAHE and denoising."""
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Resize
        resized = cv2.resize(denoised, (128, 32))
        
        # Normalize
        normalized = resized / 255.0
        
        # Convert to tensor
        tensor = torch.tensor(normalized).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    # --------------------------
    # Predict text from single line image
    # --------------------------
    def predict_line(self, line_image):
        """
        Predict text from a single line image
        
        Args:
            line_image: Preprocessed line image (numpy array or PIL Image)
            
        Returns:
            Predicted text string
        """
        try:
            # Convert PIL Image to numpy array if needed
            if hasattr(line_image, 'tobytes'):
                # PIL Image
                img_array = np.array(line_image.convert('L'))
            else:
                # Already numpy array
                img_array = line_image if isinstance(line_image, np.ndarray) else np.array(line_image)
            
            # Preprocess
            tensor = self.preprocess_line(img_array)
            
            # Predict
            with torch.no_grad():
                logits = self.model(tensor)
            
            # Decode
            pred_indices = torch.argmax(logits, dim=2)[0].cpu().numpy()
            
            # Convert indices to characters
            text = ""
            prev_idx = -1
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:  # Skip blank (0) and duplicates
                    if idx - 1 < len(self.tokenizer.chars):
                        text += self.tokenizer.chars[idx - 1]
                prev_idx = idx
            
            return text.strip()
        
        except Exception as e:
            print(f"[ERROR] predict_line failed: {e}")
            return ""

    # --------------------------
    # Full page OCR
    # --------------------------
    def predict(self, image_path):
        lines = segment_lines(image_path)

        print(f"[DEBUG] Number of detected lines = {len(lines)}") 

        if len(lines) == 0:
            return "[NO TEXT DETECTED]"

        results = []
        for line in lines:
            text = self.predict_line(line)
            if text:
                results.append(text)

        return "\n".join(results)


# STATIC FUNCTION
def recognize_image(image_path):
    recognizer = OCRRecognizer()
    return recognizer.predict(image_path)


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", default="checkpoints/ocr_finetuned_stage2_best.pth")
    args = parser.parse_args()

    recognizer = OCRRecognizer(checkpoint_path=args.checkpoint)
    text = recognizer.predict(args.image)

    print("\n===== OCR OUTPUT =====")
    print(text)
    print("======================\n")
