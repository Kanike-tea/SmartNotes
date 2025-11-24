#!/usr/bin/env python3
"""Quick evaluation script for epoch 6 model."""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Setup imports
from smartnotes.paths import setup_imports

setup_imports()

from src.dataloader.ocr_dataloader import SmartNotesOCRDataset, collate_fn
from src.model.ocr_model import CRNN

def evaluate():
    device = torch.device("cpu")  # Use CPU for stability
    print(f"Evaluating OCR model on: {device}\n")

    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = SmartNotesOCRDataset(mode="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Loaded {len(val_dataset)} validation samples\n")

    # Load model
    num_classes = len(val_dataset.tokenizer.chars)
    model = CRNN(num_classes=num_classes).to(device)
    
    ckpt_path = "checkpoints/ocr_epoch_6.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully\n")

    # Run inference
    results = []
    print("Running inference on validation set...")
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).permute(1, 0, 2)
            
            for i in range(len(labels)):
                pred_text = val_dataset.tokenizer.decode(
                    np.argmax(logits[i].cpu().numpy(), axis=1)
                )
                gt_text = val_dataset.tokenizer.decode(labels[i].cpu().numpy())
                
                if len(gt_text.strip()) == 0:
                    continue
                
                # Character Error Rate
                dist = sum(1 for a, b in zip(pred_text, gt_text) if a != b)
                cer = dist / max(len(gt_text), 1)
                
                # Word Error Rate
                pred_words = pred_text.strip().split()
                gt_words = gt_text.strip().split()
                word_dist = sum(1 for a, b in zip(pred_words, gt_words) if a != b)
                wer = word_dist / max(len(gt_words), 1)
                
                results.append((pred_text, gt_text, cer, wer))

    # Print results
    if results:
        cers = [r[2] for r in results]
        wers = [r[3] for r in results]
        
        print("\n" + "="*60)
        print("EPOCH 6 EVALUATION RESULTS")
        print("="*60)
        print(f"Total evaluated samples: {len(results)}")
        print(f"Average CER (Character Error Rate): {np.mean(cers):.4f}")
        print(f"Average WER (Word Error Rate): {np.mean(wers):.4f}")
        print(f"Median CER: {np.median(cers):.4f}")
        print(f"Median WER: {np.median(wers):.4f}")
        print(f"Min CER: {np.min(cers):.4f} | Max CER: {np.max(cers):.4f}")
        print("="*60 + "\n")

        print("Sample predictions (first 5):")
        print("-" * 60)
        for i in range(min(5, len(results))):
            pred, gt, cer, wer = results[i]
            print(f"Ground Truth: {gt}")
            print(f"Prediction:  {pred}")
            print(f"CER: {cer:.2%} | WER: {wer:.2%}")
            print("-" * 60)
        
        return np.mean(cers), np.mean(wers)
    
    return None, None

if __name__ == "__main__":
    avg_cer, avg_wer = evaluate()
    if avg_cer is not None:
        print(f"\n✓ Evaluation complete. Avg CER: {avg_cer:.4f}, Avg WER: {avg_wer:.4f}")
