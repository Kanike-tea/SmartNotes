"""Evaluation script that runs inference on a validation set.

This script is a user-invoked tool and will not be collected by pytest since it
doesn't start with `test_` and contains a `main()` guarded by a module guard.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Any
from torch.utils.data import DataLoader


def main():
    from src.dataloader.ocr_dataloader import SmartNotesOCRDataset, collate_fn
    from src.model.ocr_model import CRNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SmartNotes OCR on: {device}")

    val_dataset = SmartNotesOCRDataset(mode="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    num_classes = len(val_dataset.tokenizer.chars)
    model = CRNN(num_classes=num_classes).to(device)

    ckpt_path = "checkpoints/ocr_finetuned_stage2_best.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = []
    for imgs, labels in tqdm(val_loader, total=len(val_loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(imgs).permute(1, 0, 2)
            probs = torch.softmax(logits, dim=2).cpu().numpy()
            for i in range(len(labels)):
                pred_text = val_dataset.tokenizer.decode(np.argmax(logits[i].cpu().numpy(), axis=1))
                gt_text = val_dataset.tokenizer.decode(labels[i].cpu().numpy())
                if len(gt_text.strip()) == 0:
                    continue
                dist = sum(1 for a, b in zip(pred_text, gt_text) if a != b)
                cer = dist / max(len(gt_text), 1)
                wer = 1.0 if pred_text.strip() != gt_text.strip() else 0.0
                results.append((pred_text, gt_text, cer, wer))

    if results:
        cers = [r[2] for r in results]
        wers = [r[3] for r in results]
        print("\n================ Evaluation Summary ================")
        print(f"Total evaluated samples : {len(results)}")
        print(f"Average CER: {np.mean(cers):.4f}")
        print(f"Average WER: {np.mean(wers):.4f}")
        print("====================================================")

    for i in range(min(5, len(results))):
        pred, gt, cer, wer = results[i]
        print(f"GT : {gt}")
        print(f"PR : {pred}")
        print(f"CER: {cer:.2f} | WER: {wer:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
