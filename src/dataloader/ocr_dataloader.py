"""
OCR Dataset and Dataloader for SmartNotes project.

This module provides dataset classes and utilities for loading and preprocessing
handwritten text recognition data from multiple sources (IAM, CensusHWR, GNHK).
"""

import os
import cv2
import torch
import random
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Text Utilities
# ============================================================================

ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-() "

def clean_text(text: str) -> str:
    """
    Clean text by removing disallowed characters.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text with only allowed characters
    """
    if not text:
        return ""
    return ''.join(c for c in text if c in ALLOWED_CHARS)


# ============================================================================
# Tokenizer
# ============================================================================

class TextTokenizer:
    """
    Simple character-level tokenizer for handwritten text recognition.
    
    Handles encoding text to token indices and decoding predictions back to text.
    Compatible with CTC loss (includes blank token).
    
    Attributes:
        chars: Character vocabulary string
        blank_idx: Index reserved for CTC blank token
        char_to_idx: Mapping from character to index
        idx_to_char: Mapping from index to character
    """
    
    def __init__(self, chars: Optional[str] = None) -> None:
        """
        Initialize tokenizer with character set.
        
        Args:
            chars: Character vocabulary string. If None, uses default lowercase+digits.
        """
        if chars is None:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        
        self.chars = chars
        # CTC blank is represented by the index after all characters
        self.blank_idx = len(self.chars)
        
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """
        Encode text string to list of token indices.
        
        Args:
            text: Text to encode (will be lowercased)
            
        Returns:
            List of character indices. Unknown characters map to blank token.
        """
        if not text:
            return []
        return [self.char_to_idx.get(c, self.blank_idx) for c in text.lower()]

    def decode(self, seq: np.ndarray) -> str:
        """
        Decode token indices to text (greedy CTC decoding).
        
        Implements greedy decoding for CTC: skip blanks and collapse repeated characters.
        
        Args:
            seq: Array of token indices
            
        Returns:
            Decoded text string
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.numpy()
        
        # If blank token is present, treat this as a model output sequence and
        # apply greedy CTC decoding rules: collapse repeats and skip blanks.
        if self.blank_idx in seq:
            text = ""
            prev = None
            for idx in seq:
                if idx == self.blank_idx:
                    prev = None
                    continue
                if idx != prev:
                    text += self.idx_to_char.get(int(idx), "")
                prev = idx
            return text

        # Otherwise, treat as a direct label sequence (produced by `encode`) and
        # map indices to characters without collapsing repeats.
        text = "".join(self.idx_to_char.get(int(idx), "") for idx in seq)
        return text


# ============================================================================
# Dataset Classes
# ============================================================================

class SmartNotesOCRDataset(Dataset):
    """
    Combined OCR dataset from multiple sources.
    
    Loads and manages data from:
    - IAM Handwriting Database
    - CensusHWR Dataset
    - GNHK Dataset
    
    Automatically splits into train/val/test sets based on split ratio.
    
    Attributes:
        samples: List of (image_path, text) tuples
        tokenizer: TextTokenizer instance
        mode: 'train', 'val', or 'test' mode
    """
    
    def __init__(
        self,
        root_dir: str = "datasets",
        mode: str = 'train',
        split_ratio: float = 0.85,
        max_samples: Optional[int] = None
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing datasets
            mode: Dataset mode - 'train', 'val', or 'test'
            split_ratio: Train/val split ratio (e.g., 0.85 for 85% train)
            max_samples: Maximum number of samples to use (None for all)
            
        Raises:
            ValueError: If mode is invalid or no samples found
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{mode}'")
        
        self.root_dir = root_dir
        self.mode = mode
        self.samples: List[Tuple[str, str]] = []
        self.tokenizer = TextTokenizer()

        # Load all datasets
        logger.info(f"Loading {mode} dataset...")
        self.samples += self._load_iam()
        self.samples += self._load_census()
        self.samples += self._load_gnhk()
        self.samples += self._load_handwritten_notes()
        self.samples += self._load_printed_notes()

        total = len(self.samples)
        if total == 0:
            logger.warning("No dataset samples found. Check dataset paths.")
            raise ValueError("No samples loaded from datasets")

        # Shuffle and split
        random.shuffle(self.samples)
        split = int(total * split_ratio)
        
        if mode == 'train':
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]
        
        # Apply max samples limit if specified
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        logger.info(f"{mode.upper()} set: {len(self.samples)} samples loaded")

    def _load_iam(self) -> List[Tuple[str, str]]:
        """
        Load IAM Handwriting Database.
        
        Returns:
            List of (image_path, text) tuples
        """
        ascii_path = os.path.join(self.root_dir, "IAM/ascii/lines.txt")
        lines_dir = os.path.join(self.root_dir, "IAM/lines")
        data = []
        
        if not os.path.exists(ascii_path):
            logger.debug("IAM dataset not found.")
            return []
        
        try:
            with open(ascii_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split(" ")
                    if len(parts) < 9 or parts[1] != "ok":
                        continue
                    
                    line_id = parts[0]
                    text = clean_text(" ".join(parts[8:]))
                    if not text:
                        continue
                    
                    img_path = os.path.join(lines_dir, line_id[:3], line_id[:7], f"{line_id}.png")
                    if os.path.exists(img_path):
                        data.append((img_path, text))
        
        except Exception as e:
            logger.warning(f"Error loading IAM dataset: {e}")
        
        logger.debug(f"IAM loaded: {len(data)} samples")
        return data

    def _load_census(self) -> List[Tuple[str, str]]:
        """
        Load CensusHWR Dataset.
        
        Returns:
            List of (image_path, text) tuples
        """
        census_root = os.path.join(self.root_dir, "CensusHWR")
        data = []
        
        for split in ["train.tsv", "val.tsv", "test.tsv"]:
            tsv_path = os.path.join(census_root, split)
            if not os.path.exists(tsv_path):
                continue
            
            try:
                with open(tsv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            img_rel, label = line.split("\t")
                            text = clean_text(label)
                            if not text:
                                continue
                            
                            img_path = os.path.join(census_root, img_rel)
                            if os.path.exists(img_path):
                                data.append((img_path, text))
                        except ValueError:
                            continue
            
            except Exception as e:
                logger.warning(f"Error loading CensusHWR split {split}: {e}")
        
        logger.debug(f"CensusHWR loaded: {len(data)} samples")
        return data

    def _load_gnhk(self) -> List[Tuple[str, str]]:
        """
        Load GNHK Dataset.
        
        Returns:
            List of (image_path, text) tuples
        """
        gnhk_root = os.path.join(self.root_dir, "GNHK/test")
        data = []
        
        if not os.path.exists(gnhk_root):
            logger.debug("GNHK dataset not found.")
            return []
        
        try:
            for file in os.listdir(gnhk_root):
                if not file.endswith(".json"):
                    continue
                
                json_path = os.path.join(gnhk_root, file)
                img_path = json_path.replace(".json", ".jpg")
                
                if not os.path.exists(img_path):
                    continue
                
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        
                        if isinstance(meta, dict):
                            text = meta.get("transcription") or meta.get("text") or ""
                            if text:
                                text = clean_text(text)
                                if text:
                                    data.append((img_path, text))
                        elif isinstance(meta, list):
                            for item in meta:
                                t = item.get("transcription") or item.get("text") or ""
                                if t:
                                    text = clean_text(t)
                                    if text:
                                        data.append((img_path, text))
                
                except Exception as e:
                    logger.debug(f"Error loading JSON {file}: {e}")
        
        except Exception as e:
            logger.warning(f"Error loading GNHK dataset: {e}")
        
        logger.debug(f"GNHK loaded: {len(data)} samples")
        return data

    def _load_handwritten_notes(self) -> List[Tuple[str, str]]:
        """
        Load handwritten notes from local note files.
        
        Looks for image files (jpg, png, jpeg) in datasets/handwritten_notes_extracted directory.
        Uses OCR on extracted PDF images or direct image files.
        
        Note: These are used primarily for transfer learning/fine-tuning.
        To add text labels, create a manifest.txt file with format:
        image_path<TAB>text
        
        Returns:
            List of (image_path, text) tuples
        """
        handwritten_dir = os.path.join(self.root_dir, "handwritten_notes_extracted")
        data = []
        
        if not os.path.exists(handwritten_dir):
            logger.debug(
                f"Handwritten notes directory not found: {handwritten_dir}\n"
                f"To use handwritten notes, extract PDFs to this directory."
            )
            return []
        
        # Try to load manifest file if it exists
        manifest_path = os.path.join(self.root_dir, "handwritten_notes_manifest.txt")
        manifest = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            manifest[parts[0]] = parts[1]
            except Exception as e:
                logger.debug(f"Could not load manifest: {e}")
        
        try:
            # Find all image files recursively
            for root, dirs, files in os.walk(handwritten_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        if os.path.exists(img_path):
                            # Check manifest or skip
                            rel_path = os.path.relpath(img_path, self.root_dir)
                            if rel_path in manifest:
                                text = manifest[rel_path]
                                if text.strip():
                                    data.append((img_path, text))
        
        except Exception as e:
            logger.warning(f"Error loading handwritten notes: {e}")
        
        if len(data) > 0:
            logger.debug(f"Handwritten notes loaded: {len(data)} samples")
        else:
            logger.debug("Handwritten notes: No labeled samples found. Use manifest file for labels.")
        return data

    def _load_printed_notes(self) -> List[Tuple[str, str]]:
        """
        Load printed notes from local note files.
        
        Looks for image files (jpg, png, jpeg) in datasets/printed_notes_extracted directory.
        Uses OCR on extracted PDF images or direct image files.
        
        Note: These are used primarily for transfer learning/fine-tuning.
        To add text labels, create a manifest.txt file with format:
        image_path<TAB>text
        
        Returns:
            List of (image_path, text) tuples
        """
        printed_dir = os.path.join(self.root_dir, "printed_notes_extracted")
        data = []
        
        if not os.path.exists(printed_dir):
            logger.debug(
                f"Printed notes directory not found: {printed_dir}\n"
                f"To use printed notes, extract PDFs to this directory."
            )
            return []
        
        # Try to load manifest file if it exists
        manifest_path = os.path.join(self.root_dir, "printed_notes_manifest.txt")
        manifest = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            manifest[parts[0]] = parts[1]
            except Exception as e:
                logger.debug(f"Could not load manifest: {e}")
        
        try:
            # Find all image files recursively
            for root, dirs, files in os.walk(printed_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        if os.path.exists(img_path):
                            # Check manifest or skip
                            rel_path = os.path.relpath(img_path, self.root_dir)
                            if rel_path in manifest:
                                text = manifest[rel_path]
                                if text.strip():
                                    data.append((img_path, text))
        
        except Exception as e:
            logger.warning(f"Error loading printed notes: {e}")
        
        if len(data) > 0:
            logger.debug(f"Printed notes loaded: {len(data)} samples")
        else:
            logger.debug("Printed notes: No labeled samples found. Use manifest file for labels.")
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor) where:
            - image_tensor: (1, 32, 128) grayscale image tensor
            - label_tensor: token indices for the text
        """
        img_path, text = self.samples[idx]
        
        # Read and preprocess image
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                # Return blank image
                img = np.zeros((32, 128), dtype=np.uint8)
            
            img = cv2.resize(img, (128, 32))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0).float()
        
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            img = torch.zeros(1, 32, 128)
        
        # Encode label
        label = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        return img, label

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-length sequences.
    
    Pads images and labels to the maximum dimensions in the batch.
    
    Args:
        batch: List of (image, label) tuples
        
    Returns:
        Tuple of (padded_images, padded_labels) tensors
    """
    imgs, labels = zip(*batch)
    
    # Pad images to max width (height should be constant)
    max_w = max(img.shape[-1] for img in imgs)
    imgs_padded = [
        torch.nn.functional.pad(i, (0, max_w - i.shape[-1])) 
        for i in imgs
    ]
    
    # Pad labels to max length
    max_l = max(len(lbl) for lbl in labels)
    labels_padded = [
        torch.nn.functional.pad(l, (0, max_l - len(l))) 
        for l in labels
    ]
    
    return torch.stack(imgs_padded), torch.stack(labels_padded)
 
