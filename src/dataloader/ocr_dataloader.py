import os
import cv2
import torch
import random
import json
from torch.utils.data import Dataset
import numpy as np

# -----------------------------
# Text cleaner
# -----------------------------
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-() "

def clean_text(text):
    return ''.join(c for c in text if c in ALLOWED_CHARS)

# -----------------------------
# Tokenizer
# -----------------------------
class TextTokenizer:
    def __init__(self):
        # Add all allowed chars
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        
        # Add explicit blank token for CTC (no visible char)
        self.blank_idx = len(self.chars)
        
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        # Encode text -> indices, unknowns map to blank
        return [self.char_to_idx.get(c, self.blank_idx) for c in text.lower()]

    def decode(self, seq):
        # Greedy CTC decoding (skip blanks + collapse repeats)
        text = ""
        prev = None
        for idx in seq:
            if idx == self.blank_idx:
                prev = None  # reset repeat tracker after a blank
                continue
            if idx != prev:
                text += self.idx_to_char.get(idx, "")
            prev = idx
        return text

# -----------------------------
# Dataset class
# -----------------------------
class SmartNotesOCRDataset(Dataset):
    def __init__(self, root_dir="datasets", mode='train', split_ratio=0.85):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []
        self.tokenizer = TextTokenizer()

        # Load all datasets
        self.samples += self._load_iam()
        self.samples += self._load_census()
        self.samples += self._load_gnhk()

        total = len(self.samples)
        if total == 0:
            print("No dataset samples found. Check paths.")
            return

        random.shuffle(self.samples)
        split = int(total * split_ratio)
        if mode == 'train':
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]

        print(f"{mode.upper()} Loaded: {len(self.samples)} samples")

    # IAM
    def _load_iam(self):
        ascii_path = os.path.join(self.root_dir, "IAM/ascii/lines.txt")
        lines_dir = os.path.join(self.root_dir, "IAM/lines")
        data = []
        if not os.path.exists(ascii_path):
            print("IAM dataset not found.")
            return []
        with open(ascii_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split(" ")
                if len(parts) < 9 or parts[1] != "ok": continue
                line_id = parts[0]
                text = clean_text(" ".join(parts[8:]))
                img_path = os.path.join(lines_dir, line_id[:3], line_id[:7], f"{line_id}.png")
                if os.path.exists(img_path):
                    data.append((img_path, text))
        print(f"IAM Loaded: {len(data)}")
        return data

    # CensusHWR
    def _load_census(self):
        census_root = os.path.join(self.root_dir, "CensusHWR")
        data = []
        for split in ["train.tsv", "val.tsv", "test.tsv"]:
            tsv_path = os.path.join(census_root, split)
            if not os.path.exists(tsv_path): continue
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        img_rel, label = line.split("\t")
                        img_path = os.path.join(census_root, img_rel)
                        if os.path.exists(img_path):
                            data.append((img_path, clean_text(label)))
                    except ValueError:
                        continue
        print(f"CensusHWR Loaded: {len(data)}")
        return data

    # GNHK
    def _load_gnhk(self):
        gnhk_root = os.path.join(self.root_dir, "GNHK/test")
        data = []
        if not os.path.exists(gnhk_root):
            print("GNHK dataset not found.")
            return []
        for file in os.listdir(gnhk_root):
            if file.endswith(".json"):
                json_path = os.path.join(gnhk_root, file)
                img_path = json_path.replace(".json", ".jpg")
                if not os.path.exists(img_path): continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        if isinstance(meta, dict):
                            text = meta.get("transcription") or meta.get("text") or ""
                            if text:
                                data.append((img_path, clean_text(text)))
                        elif isinstance(meta, list):
                            for item in meta:
                                t = item.get("transcription") or item.get("text") or ""
                                if t:
                                    data.append((img_path, clean_text(t)))
                except Exception:
                    continue
        print(f"GNHK Loaded: {len(data)}")
        return data

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).float()
        label = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.samples)

# Collate fn
def collate_fn(batch):
    imgs, labels = zip(*batch)
    max_w = max(img.shape[-1] for img in imgs)
    max_l = max(len(lbl) for lbl in labels)
    imgs = [torch.nn.functional.pad(i, (0, max_w - i.shape[-1])) for i in imgs]
    labels = [torch.nn.functional.pad(l, (0, max_l - len(l))) for l in labels]
    return torch.stack(imgs), torch.stack(labels)
