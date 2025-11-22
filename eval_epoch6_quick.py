#!/usr/bin/env python3
"""Quick evaluation script for epoch 6 model - samples only."""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Add to path
sys.path.insert(0, '/Users/kanike/Desktop/SmartNotes/SmartNotes')

from src.dataloader.ocr_dataloader import SmartNotesOCRDataset, collate_fn
from src.model.ocr_model import CRNN

def evaluate(sample_size=5000):
    device = torch.device("cpu")
    print(f"Evaluating OCR model on: {device}\n")

    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = SmartNotesOCRDataset(mode="val")
    full_size = len(val_dataset)
    print(f"Full dataset size: {full_size} samples")
    
    # Sample subset for quick evaluation
    if sample_size and full_size > sample_size:
        indices = np.random.choice(full_size, size=sample_size, replace=False)
        val_dataset = Subset(val_dataset, indices)
        print(f"Sampling {sample_size} random samples for evaluation\n")
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Load model
    num_classes = len(val_dataset.dataset.tokenizer.chars)
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
    print("Running inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).permute(1, 0, 2)
            
            for i in range(len(labels)):
                pred_text = val_dataset.dataset.tokenizer.decode(
                    np.argmax(logits[i].cpu().numpy(), axis=1)
                )
                gt_text = val_dataset.dataset.tokenizer.decode(labels[i].cpu().numpy())
                
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
        cers = np.array([r[2] for r in results])
        wers = np.array([r[3] for r in results])
        
        print("\n" + "="*60)
        print("EPOCH 6 EVALUATION RESULTS")
        print("="*60)
        print(f"Evaluated samples: {len(results)}")
        print(f"Average CER: {np.mean(cers):.4f} (±{np.std(cers):.4f})")
        print(f"Average WER: {np.mean(wers):.4f} (±{np.std(wers):.4f})")
        print(f"Median CER: {np.median(cers):.4f}")
        print(f"Median WER: {np.median(wers):.4f}")
        print(f"Best CER: {np.min(cers):.4f} | Worst CER: {np.max(cers):.4f}")
        print("="*60)
        
        # Analysis
        perfect = sum(1 for cer in cers if cer == 0)
        excellent = sum(1 for cer in cers if 0 < cer <= 0.05)
        good = sum(1 for cer in cers if 0.05 < cer <= 0.15)
        fair = sum(1 for cer in cers if 0.15 < cer <= 0.30)
        poor = sum(1 for cer in cers if cer > 0.30)
        
        print("\nCER Distribution:")
        print(f"  Perfect (0.00):           {perfect:4d} ({100*perfect/len(cers):5.2f}%)")
        print(f"  Excellent (0-5%):         {excellent:4d} ({100*excellent/len(cers):5.2f}%)")
        print(f"  Good (5-15%):             {good:4d} ({100*good/len(cers):5.2f}%)")
        print(f"  Fair (15-30%):            {fair:4d} ({100*fair/len(cers):5.2f}%)")
        print(f"  Poor (>30%):              {poor:4d} ({100*poor/len(cers):5.2f}%)")
        print("="*60 + "\n")

        print("Sample predictions (first 5):")
        print("-" * 60)
        for i in range(min(5, len(results))):
            pred, gt, cer, wer = results[i]
            print(f"GT: {gt}")
            print(f"PR: {pred}")
            print(f"CER: {cer:.2%} | WER: {wer:.2%}")
            print()
        
        return np.mean(cers), np.mean(wers)
    
    return None, None

if __name__ == "__main__":
    avg_cer, avg_wer = evaluate(sample_size=5000)
    if avg_cer is not None:
        print(f"\n✓ Evaluation complete")
        print(f"  Average CER: {avg_cer:.4f}")
        print(f"  Average WER: {avg_wer:.4f}")
        print(f"\nRecommendation: {'Continue training (train 14+ more epochs)' if avg_cer > 0.10 else 'Model quality is good, further training optional'}")
    else:
        print("Evaluation failed")
