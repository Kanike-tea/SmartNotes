#!/usr/bin/env python3
"""Fast evaluation script for epoch 6 model - minimal dataset loading."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, '/Users/kanike/Desktop/SmartNotes/SmartNotes')

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer
import cv2

def load_single_image(img_path: str) -> Optional[torch.Tensor]:
    """Load and preprocess single image."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (128, 32))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).float()
    except Exception as e:
        return None

def quick_eval():
    """Quick evaluation on small test set."""
    device = torch.device("cpu")
    print(f"Quick Evaluation on: {device}\n")
    
    # Load model
    print("Loading model...")
    tokenizer = TextTokenizer()
    num_classes = len(tokenizer.chars)
    model = CRNN(num_classes=num_classes).to(device)
    
    ckpt_path = "checkpoints/ocr_epoch_6.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully\n")
    
    # Find a few test images from IAM dataset
    iam_dir = "datasets/IAM/ascii"
    if not os.path.exists(iam_dir):
        print(f"ERROR: IAM dataset not found at {iam_dir}")
        return
    
    # Collect test samples
    print("Scanning for test images...")
    test_samples = []
    
    for root, dirs, files in os.walk(iam_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                try:
                    with open(txt_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split()
                            if len(parts) >= 8:
                                status = parts[1]
                                if status == "ok":
                                    img_rel_path = parts[0].replace("-", "/") + ".png"
                                    text = " ".join(parts[8:])
                                    img_path = os.path.join(root, "..", "lines", img_rel_path)
                                    if os.path.exists(img_path):
                                        test_samples.append((img_path, text))
                                        if len(test_samples) >= 500:
                                            break
                except Exception:
                    pass
            if len(test_samples) >= 500:
                break
        if len(test_samples) >= 500:
            break
    
    if not test_samples:
        print("ERROR: No test samples found")
        return
    
    print(f"Found {len(test_samples)} test samples\n")
    print("Running inference...")
    
    results = []
    with torch.no_grad():
        for img_path, gt_text in tqdm(test_samples, desc="Evaluating"):
            img = load_single_image(img_path)
            if img is None:
                continue
            
            img = img.to(device)
            logits = model(img.unsqueeze(0)).permute(1, 0, 2)
            pred_indices = torch.argmax(logits[:, 0, :], dim=1)
            pred_text = tokenizer.decode(pred_indices.cpu().numpy())
            
            # Normalize ground truth
            gt_text = gt_text.lower()
            
            if len(gt_text.strip()) > 0:
                dist = sum(1 for a, b in zip(pred_text, gt_text) if a != b)
                cer = dist / max(len(gt_text), 1)
                
                pred_words = pred_text.strip().split()
                gt_words = gt_text.strip().split()
                word_dist = sum(1 for a, b in zip(pred_words, gt_words) if a != b)
                wer = word_dist / max(len(gt_words), 1)
                
                results.append((pred_text, gt_text, cer, wer))
    
    if not results:
        print("No valid results")
        return
    
    # Print results
    cers = np.array([r[2] for r in results])
    wers = np.array([r[3] for r in results])
    
    print("\n" + "="*70)
    print("EPOCH 6 QUICK EVALUATION - IAM TEST SET")
    print("="*70)
    print(f"Evaluated samples: {len(results)}")
    print(f"Average CER: {np.mean(cers):.4f} (±{np.std(cers):.4f})")
    print(f"Average WER: {np.mean(wers):.4f} (±{np.std(wers):.4f})")
    print(f"Best CER: {np.min(cers):.4f} | Worst CER: {np.max(cers):.4f}")
    print("="*70)
    
    # Distribution
    perfect = sum(1 for c in cers if c == 0)
    excellent = sum(1 for c in cers if 0 < c <= 0.05)
    good = sum(1 for c in cers if 0.05 < c <= 0.15)
    fair = sum(1 for c in cers if 0.15 < c <= 0.30)
    poor = sum(1 for c in cers if c > 0.30)
    
    print("\nCER Distribution:")
    print(f"  Perfect (0%):      {perfect:5d} ({100*perfect/len(cers):5.2f}%)")
    print(f"  Excellent (0-5%):  {excellent:5d} ({100*excellent/len(cers):5.2f}%)")
    print(f"  Good (5-15%):      {good:5d} ({100*good/len(cers):5.2f}%)")
    print(f"  Fair (15-30%):     {fair:5d} ({100*fair/len(cers):5.2f}%)")
    print(f"  Poor (>30%):       {poor:5d} ({100*poor/len(cers):5.2f}%)")
    print("="*70)
    
    # Show samples
    print(f"\nSample predictions (first 5):")
    print("-" * 70)
    for i in range(min(5, len(results))):
        pred, gt, cer, wer = results[i]
        print(f"GT: {gt}")
        print(f"PR: {pred}")
        print(f"CER: {cer:.2%} | WER: {wer:.2%}")
        print()
    
    print(f"✓ Evaluation complete")
    print(f"  Average CER: {np.mean(cers):.4f}")
    print(f"  Average WER: {np.mean(wers):.4f}\n")

if __name__ == "__main__":
    quick_eval()
