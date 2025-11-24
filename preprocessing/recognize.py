# recognize.py — FINAL FULL-PAGE OCR VERSION

import sys
import platform
import argparse
from pathlib import Path

# Setup imports using path utilities
from smartnotes.paths import setup_imports, get_checkpoint_dir

setup_imports()

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
    def __init__(self, checkpoint_path="checkpoints/ocr_epoch_6.pth"):

        self.device = torch.device("cpu")
        self.tokenizer = TextTokenizer()
        self.num_classes = len(self.tokenizer.chars)

        # Load CRNN model
        self.model = CRNN(num_classes=self.num_classes).to(self.device)

        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Handle both wrapped and unwrapped checkpoints
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
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
    # Preprocess single line (9-step enhanced pipeline)
    # --------------------------
    def preprocess_line(self, img):
        """
        Enhanced 9-step preprocessing pipeline for robustness
        Handles various image qualities and text types
        """
        try:
            # Step 1: Ensure grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Calculate statistics for adaptive processing
            h, w = img.shape
            mean_val = np.mean(img)
            std_val = np.std(img)
            
            # Step 3: Intermediate resize (preserve aspect ratio)
            # This is critical - don't directly resize to 128x32
            target_height = 64
            scale = target_height / h
            new_width = max(20, int(w * scale))
            img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Step 4: Adaptive CLAHE based on contrast
            clip_limit = 2.0
            if std_val < 30:
                clip_limit = 3.0  # Low contrast - enhance more
            elif std_val > 60:
                clip_limit = 1.5  # High contrast - enhance less
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_resized)
            
            # Step 5: Conditional denoising (only if noisy)
            if std_val > 50:
                denoised = cv2.fastNlMeansDenoising(enhanced, h=15, templateWindowSize=7, searchWindowSize=21)
            else:
                denoised = enhanced
            
            # Step 6: Sharpening (critical for printed text)
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Step 7: Binarization
            binary = cv2.adaptiveThreshold(sharpened, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Step 8: Final resize to model size (128x32)
            final = cv2.resize(binary, (128, 32), interpolation=cv2.INTER_AREA)
            
            # Step 9: Normalize and tensorize
            normalized = final.astype(np.float32) / 255.0
            tensor = torch.tensor(normalized).float().unsqueeze(0).unsqueeze(0)
            
            return tensor.to(self.device)
        
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            # Fallback: simple preprocessing
            normalized = cv2.resize(img, (128, 32)) / 255.0
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
            # Convert to numpy array if needed
            if isinstance(line_image, np.ndarray):
                # Already numpy array
                img_array = line_image
            else:
                # Try PIL Image conversion
                try:
                    img_array = np.array(line_image.convert('L'))
                except:
                    # Fallback - just convert to array
                    img_array = np.array(line_image)
            
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
    # Text validation
    # --------------------------
    def _is_valid_text(self, text):
        """
        Validate text quality to filter garbage output
        
        Args:
            text: Text string to validate
            
        Returns:
            True if text appears valid, False otherwise
        """
        clean = text.strip()
        
        if len(clean) == 0:
            return False
        
        # Count alphanumeric characters
        alphanum_count = sum(c.isalnum() for c in clean)
        
        # Must have some alphanumeric content
        if alphanum_count == 0:
            return False
        
        # Must have reasonable ratio of alphanumeric to total
        if alphanum_count / len(clean) < 0.3:
            return False
        
        # Must not be repetitive garbage (e.g., "aaaaaa" or "!!!!!!!")
        if len(set(clean)) < 3:
            return False
        
        return True

    # --------------------------
    # Full page OCR with validation
    # --------------------------
    def predict(self, image_path, debug=False):
        """
        Predict text from full page image
        
        Args:
            image_path: Path to image file
            debug: If True, print debug information
            
        Returns:
            Extracted text or error message
        """
        lines = segment_lines(image_path, debug=debug)

        if len(lines) == 0:
            return "[NO TEXT DETECTED - SEGMENTATION FAILED]"

        results = []
        for i, line in enumerate(lines):
            # Validate line image
            if line is None or line.size == 0:
                if debug:
                    print(f"[DEBUG] Skipping invalid line {i}")
                continue
            
            # Check dimensions
            h, w = line.shape
            if h < 10 or w < 20:
                if debug:
                    print(f"[DEBUG] Skipping small line {i}: {w}x{h}")
                continue
            
            # Recognize text
            text = self.predict_line(line)
            
            # Filter and validate
            if text and len(text.strip()) > 0:
                text = ' '.join(text.split())  # Clean whitespace
                
                if self._is_valid_text(text):
                    results.append(text)
                    if debug:
                        print(f"[DEBUG] Line {i}: {text}")
                else:
                    if debug:
                        print(f"[DEBUG] Filtered line {i}: {text}")
        
        if len(results) == 0:
            return "[NO TEXT DETECTED - RECOGNITION FAILED]"
        
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    recognizer = OCRRecognizer(checkpoint_path=args.checkpoint)
    text = recognizer.predict(args.image, debug=args.debug)

    print("\n===== OCR OUTPUT =====")
    print(text)
    print("======================\n")
