import torch
from ocr_dataloader import SmartNotesOCRDataset
from ocr_model import CRNN

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Testing on: {device}")

# -----------------------------
# Dataset and model load
# -----------------------------
dataset = SmartNotesOCRDataset(mode='val')
num_classes = len(dataset.tokenizer.chars)

model = CRNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/ocr_epoch_20.pth", map_location=device))
model.eval()

# -----------------------------
# Run sample predictions
# -----------------------------
with torch.no_grad():
    displayed = 0
    idx = 0
    # iterate through the dataset until we display 5 valid samples or exhaust the dataset
    while displayed < 5 and idx < len(dataset):
        img, label = dataset[idx]
        idx += 1

        # skip missing images
        if img is None:
            print(f"Skipping sample {idx}: image is None")
            continue

        # ensure img is a torch tensor
        if not torch.is_tensor(img):
            try:
                img = torch.tensor(img)
            except Exception:
                print(f"Skipping sample {idx}: image is not a tensor and cannot be converted")
                continue

        img = img.unsqueeze(0).to(device)
        preds = model(img)
        preds = preds.permute(1, 0, 2).cpu()

        seq = torch.argmax(preds[0], dim=1).numpy()
        pred_text = dataset.tokenizer.decode(seq)
        gt_text = dataset.tokenizer.decode(label.numpy())

        print(f"\nSample {displayed + 1}")
        print(f"Predicted: {pred_text}")
        print(f"Ground Truth: {gt_text}")
        print("-" * 60)

        displayed += 1

    if displayed == 0:
        print("No valid samples found in the dataset to display.")
