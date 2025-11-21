import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.ocr_model import CRNN
from dataloader.ocr_dataloader import SmartNotesOCRDataset, collate_fn


# ================================================================
# Configuration
# ================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Fine-tuning SmartNotes OCR (Stage 2) on: {device}")
print("Note: CTC Loss will still fallback to CPU for some operations.")

epochs = 20
batch_size = 16
learning_rate = 1e-4
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

base_ckpt = os.path.join(save_dir, "ocr_finetuned_epoch_20.pth")
latest_ckpt = os.path.join(save_dir, "ocr_finetuned_stage2_latest.pth")
best_ckpt = os.path.join(save_dir, "ocr_finetuned_stage2_best.pth")

# ================================================================
# Data
# ================================================================
train_dataset = SmartNotesOCRDataset(mode='train')
val_dataset = SmartNotesOCRDataset(mode='val')

# Use subset for faster fine-tuning
train_dataset.samples = train_dataset.samples[:10000]
val_dataset.samples = val_dataset.samples[:2000]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
num_classes = len(train_dataset.tokenizer.chars)  # Get from dataset

print(f"TRAIN Loaded: {len(train_dataset.samples)} samples")
print(f"VAL Loaded: {len(val_dataset.samples)} samples")

# ================================================================
# Model Setup
# ================================================================
model = CRNN(num_classes=num_classes).to(device)
criterion = nn.CTCLoss(blank=num_classes, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_epoch = 1
best_val_loss = float("inf")

# ================================================================
# Load Checkpoint (Resume or Base)
# ================================================================
if os.path.exists(latest_ckpt):
    print(f"Resuming training from {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resumed from epoch {start_epoch - 1}")
elif os.path.exists(base_ckpt):
    print(f"Loading pretrained weights from {base_ckpt}")
    state_dict = torch.load(base_ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded base checkpoint successfully.")
    except RuntimeError:
        print("Size mismatch detected — loading compatible layers only.")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        skipped = [k for k in state_dict.keys() if k not in pretrained_dict]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} layers, skipped {len(skipped)}")
else:
    print("No pretrained checkpoint found — starting from scratch.")

# ================================================================
# Training Loop
# ================================================================
for epoch in range(start_epoch, epochs + 1):
    print(f"\nEpoch [{epoch}/{epochs}] — Fine-tuning...")
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        input_lengths = torch.full((imgs.size(0),), imgs.size(3) // 4, dtype=torch.long)
        target_lengths = torch.tensor([torch.count_nonzero(lbl).item() for lbl in labels], dtype=torch.long)

        outputs = model(imgs)
        log_probs = outputs.log_softmax(2)
        
        # Move CTC inputs to CPU since MPS doesn't support CTC
        loss = criterion(log_probs.cpu(), labels.cpu(), input_lengths.cpu(), target_lengths.cpu())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ================================================================
    # Validation
    # ================================================================
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            input_lengths = torch.full((imgs.size(0),), imgs.size(3) // 4, dtype=torch.long)
            target_lengths = torch.tensor([torch.count_nonzero(lbl).item() for lbl in labels], dtype=torch.long)

            outputs = model(imgs)
            log_probs = outputs.log_softmax(2)
            loss = criterion(log_probs.cpu(), labels.cpu(), input_lengths.cpu(), target_lengths.cpu())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch}/{epochs}] Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # ================================================================
    # Save Latest + Best
    # ================================================================
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }, latest_ckpt)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_ckpt)
        print(f"New best model saved: {best_ckpt}")

    # ================================================================
    # Sample Predictions
    # ================================================================
    # Show sample predictions using the dataset's tokenizer
    with torch.no_grad():
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        preds = preds.permute(1, 0, 2).cpu()

        print("\nSample Predictions:")
        for i in range(3):
            seq = torch.argmax(preds[i], dim=1).numpy()
            pred_text = train_dataset.tokenizer.decode(seq)
            gt_text = train_dataset.tokenizer.decode(labels[i].cpu().numpy())
            print(f"Predicted: {pred_text}")
            print(f"Ground Truth: {gt_text}")
            print("-" * 40)
