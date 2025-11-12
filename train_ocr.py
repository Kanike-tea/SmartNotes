import os
# Ensure MPS fallback for ops (like CTC) is enabled before importing torch
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
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Allow CTC loss fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training SmartNotes OCR on: {device}")
if device.type == "mps":
    print("Note: running on MPS. Some ops (e.g. CTC) may fall back to CPU if not implemented on MPS.")

# -----------------------------
# 2. Datasets and Dataloaders
# -----------------------------
train_dataset = SmartNotesOCRDataset(mode='train')
val_dataset = SmartNotesOCRDataset(mode='val')

# Use manageable subset for faster training
train_dataset.samples = train_dataset.samples[:20000]
val_dataset.samples = val_dataset.samples[:5000]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 3. Model, Loss, Optimizer
# -----------------------------
num_classes = len(train_dataset.tokenizer.chars)
model = CRNN(num_classes=num_classes).to(device)

criterion = nn.CTCLoss(blank=num_classes, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# AMP scaler for CUDA; ignored for MPS
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# -----------------------------
# 4. Training Loop
# -----------------------------
num_epochs = 20
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}] — Training...")
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)

        input_lengths = torch.full((imgs.size(0),), imgs.size(3) // 4, dtype=torch.long)
        target_lengths = torch.tensor([torch.count_nonzero(lbl).item() for lbl in labels], dtype=torch.long)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds.log_softmax(2), labels, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

            if torch.cuda.is_available():
                autocast_ctx = torch.cuda.amp.autocast()
            elif device.type == "mps" and hasattr(torch, "autocast"):
                autocast_ctx = torch.autocast(device_type="mps")
            else:
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                preds = model(imgs)
                loss = criterion(preds.log_softmax(2), labels, input_lengths, target_lengths)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # -----------------------------
    # Show Predictions
    # -----------------------------
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
        ckpt_path = os.path.join(save_dir, f"ocr_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

print("\nTraining complete! Model ready for inference or fine-tuning.")