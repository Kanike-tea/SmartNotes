import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Any
from torch.utils.data import DataLoader
from ocr_dataloader import SmartNotesOCRDataset, collate_fn
from ocr_model import CRNN

# Optional LM decoding (pyctcdecode)
HAS_LM = False
decoder: Optional[Any] = None
build_ctcdecoder = None

try:
    from pyctcdecode.decoder import build_ctcdecoder as _build_ctcdecoder
    build_ctcdecoder = _build_ctcdecoder
    HAS_LM = True
except ImportError:
    print("pyctcdecode not found — running greedy decoding only.\nInstall it via: pip install pyctcdecode kenlm\n")

# ================================================================
# 1. Device setup
# ================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Testing SmartNotes OCR on: {device}")

# ================================================================
# 2. Dataset setup
# ================================================================
val_dataset = SmartNotesOCRDataset(mode="val")
val_dataset.samples = val_dataset.samples[:2000]  # limit for quick eval
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
print(f"Loaded {len(val_dataset.samples)} validation samples.")

# ================================================================
# 3. Model setup
# ================================================================
num_classes = len(val_dataset.tokenizer.chars)
model = CRNN(num_classes=num_classes).to(device)

ckpt_path = "checkpoints/ocr_finetuned_stage2_best.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded model from {ckpt_path}")

# ================================================================
# 4. Optional LM Decoder
# ================================================================
if HAS_LM:
    assert build_ctcdecoder is not None
    vocab = list(val_dataset.tokenizer.chars) + ["-"]
    decoder = build_ctcdecoder(labels=vocab)
    print("Language model decoder initialized (beam search mode).")
else:
    decoder = None

# ================================================================
# 5. Evaluation
# ================================================================
results = []
print("Evaluating:")
for imgs, labels in tqdm(val_loader, total=len(val_loader)):
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        preds = model(imgs).permute(1, 0, 2).cpu()  # (B, T, C)
        probs = torch.softmax(preds, dim=2).numpy()

        for i in range(len(labels)):
            if HAS_LM:
                assert decoder is not None
                pred_text = decoder.decode(np.log(probs[i] + 1e-8))
            else:
                seq = torch.argmax(preds[i], dim=1).numpy()
                pred_text = val_dataset.tokenizer.decode(seq)

            gt_text = val_dataset.tokenizer.decode(labels[i].cpu().numpy())

            # Compute CER and WER
            if len(gt_text.strip()) == 0:
                continue
            dist = sum(1 for a, b in zip(pred_text, gt_text) if a != b)
            cer = dist / max(len(gt_text), 1)
            wer = 1.0 if pred_text.strip() != gt_text.strip() else 0.0

            results.append((pred_text, gt_text, cer, wer))

# ================================================================
# 6. Summary metrics
# ================================================================
cers = [r[2] for r in results]
wers = [r[3] for r in results]

avg_cer = np.mean(cers)
avg_wer = np.mean(wers)

print("\n================ Evaluation Summary ================")
print(f"Total evaluated samples : {len(results)}")
print(f"Average CER (Character Error Rate): {avg_cer:.4f}")
print(f"Average WER (Word Error Rate): {avg_wer:.4f}")
print("====================================================")

# ================================================================
# 7. Show sample predictions
# ================================================================
print("\nSample Predictions:\n")
for i in range(5):
    pred, gt, cer, wer = results[i]
    print(f"GT : {gt}")
    print(f"PR : {pred}")
    print(f"CER: {cer:.2f} | WER: {wer:.2f}")
    print("-" * 50)

# ================================================================
# 8. Save results
# ================================================================
import pandas as pd
out_path = "ocr_stage2_predictions.csv"
pd.DataFrame(results, columns=["Predicted", "Ground Truth", "CER", "WER"]).to_csv(out_path, index=False)
print(f"Saved detailed results to: {out_path}")
