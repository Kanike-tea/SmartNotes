import os
# Ensure MPS fallback for CTC loss (important for macOS)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import contextlib

from ocr_dataloader import SmartNotesOCRDataset, collate_fn
from ocr_model import CRNN

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Fine-tuning SmartNotes OCR on: {device}")
if device.type == "mps":
    print("Note: CTC Loss will still fallback to CPU for some operations.")

# -----------------------------
# 2. Dataset setup
# -----------------------------
train_dataset = SmartNotesOCRDataset(mode='train')
val_dataset = SmartNotesOCRDataset(mode='val')

train_dataset.samples = train_dataset.samples[:30000]  # moderate fine-tuning subset
val_dataset.samples = val_dataset.samples[:8000]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 3. Model, Loss, Optimizer
# -----------------------------
num_classes = len(train_dataset.tokenizer.chars)
model = CRNN(num_classes=num_classes).to(device)

criterion = nn.CTCLoss(blank=num_classes, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -----------------------------
# 4. Safe checkpoint loading
# -----------------------------
best_ckpt = "checkpoints/ocr_best.pth"
if os.path.exists(best_ckpt):
    print(f"Loading checkpoint safely from: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model_state = model.state_dict()

    # Keep only layers that match shape and name
    compatible_weights = {k: v for k, v in checkpoint.items()
                          if k in model_state and v.shape == model_state[k].shape}

    mismatched = [k for k in checkpoint.keys() if k not in compatible_weights]
    if mismatched:
        print(f"Skipping {len(mismatched)} mismatched layers: {mismatched}")

    model_state.update(compatible_weights)
    model.load_state_dict(model_state)
    print(f"Loaded {len(compatible_weights)}/{len(model_state)} layers successfully.")
else:
    print("No existing checkpoint found — starting fine-tuning from scratch.")

# -----------------------------
# 5. Fine-tuning loop
# -----------------------------
num_epochs = 20
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}] — Fine-tuning...")
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)

        input_lengths = torch.full((imgs.size(0),), imgs.size(3) // 4, dtype=torch.long)
        target_lengths = torch.tensor([torch.count_nonzero(lbl).item() for lbl in labels], dtype=torch.long)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds.log_softmax(2), labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            input_lengths = torch.full((imgs.size(0),), imgs.size(3) // 4, dtype=torch.long)
            target_lengths = torch.tensor([torch.count_nonzero(lbl).item() for lbl in labels], dtype=torch.long)

            with contextlib.nullcontext():
                preds = model(imgs)
                loss = criterion(preds.log_softmax(2), labels, input_lengths, target_lengths)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Show few predictions
    model.eval()
    imgs, labels = next(iter(val_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(imgs)
        preds = preds.permute(1, 0, 2).cpu()

        print("\nSample Predictions:")
        for i in range(3):
            seq = torch.argmax(preds[i], dim=1).numpy()
            pred_text = val_dataset.tokenizer.decode(seq)
            gt_text = val_dataset.tokenizer.decode(labels[i].cpu().numpy())
            print(f"Predicted: {pred_text}")
            print(f"Ground Truth: {gt_text}")
            print("----------------------------------------")

    # Save every 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join(save_dir, f"ocr_finetuned_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved fine-tuned checkpoint: {ckpt_path}")

print("\nFine-tuning complete! Updated model saved for inference.")
