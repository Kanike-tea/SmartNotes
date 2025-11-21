"""
Step-by-step guide to include handwritten and printed notes in OCR training.
"""

# ============================================================================
# HANDWRITTEN AND PRINTED NOTES INTEGRATION GUIDE
# ============================================================================

## Overview

SmartNotes can now train on your handwritten and printed notes. The process involves:

1. **Installing PDF extraction tools**
2. **Extracting images from PDFs** 
3. **Running training with the new datasets**

## Step 1: Install Required Dependencies

```bash
# Install pdf2image and poppler (required for PDF extraction)

# On macOS:
brew install poppler
pip install -r requirements.txt

# On Linux (Ubuntu/Debian):
sudo apt-get install poppler-utils
pip install -r requirements.txt

# On Windows:
# Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
# Or use: choco install poppler (if using Chocolatey)
pip install -r requirements.txt
```

## Step 2: Extract Images from PDFs

The project includes a PDF processor that converts PDF pages into images suitable for OCR training.

### Quick Start - Extract All PDFs at Once

```bash
# Extract handwritten notes
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes" \
  --output datasets/handwritten_notes_extracted \
  --dpi 150

# Extract printed notes
python -m src.dataloader.pdf_processor \
  --input "datasets/printed notes" \
  --output datasets/printed_notes_extracted \
  --dpi 150
```

### Advanced Options

```bash
# Extract with higher quality (slower, uses more disk)
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes" \
  --output datasets/handwritten_notes_extracted \
  --dpi 300

# Extract and automatically segment text regions
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes" \
  --output datasets/handwritten_notes_extracted \
  --extract-regions
```

### Understanding Output

The extracted images are organized as follows:

```
datasets/handwritten_notes_extracted/
  ├── BCS401-module-1-writtten/
  │   ├── BCS401-module-1-writtten_page000.png
  │   ├── BCS401-module-1-writtten_page001.png
  │   └── ...
  ├── ada/
  │   ├── ada_page000.png
  │   ├── ada_page001.png
  │   └── ...
  └── ...

datasets/printed_notes_extracted/
  ├── BCS401-module-2-textbook/
  │   ├── BCS401-module-2-textbook_page000.png
  │   └── ...
  └── ...
```

## Step 3: Prepare Metadata (Optional but Recommended)

For better training, you should provide ground truth text for each image. Create a manifest file:

### Option A: Automatic with OCR

The dataloader can use OCR predictions as weak labels:

```python
from preprocessing.recognize import OCRRecognizer

recognizer = OCRRecognizer()

# Generate labels for extracted notes
import os
from pathlib import Path

notes_dir = "datasets/handwritten_notes_extracted"
output_manifest = "datasets/handwritten_notes_manifest.txt"

with open(output_manifest, 'w') as f:
    for root, dirs, files in os.walk(notes_dir):
        for file in sorted(files):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                text = recognizer.recognize(img_path)
                f.write(f"{img_path}\t{text}\n")
```

### Option B: Manual Annotation

Create a manifest file manually with image paths and transcriptions:

```txt
datasets/handwritten_notes_extracted/ada/ada_page000.png	Introduction to Algorithms
datasets/handwritten_notes_extracted/ada/ada_page001.png	Analysis of recursive algorithms
datasets/printed_notes_extracted/ada/BCS401_page000.png	Chapter 1: Fundamentals
...
```

### Option C: Semi-automatic with OCR

Use OCR predictions and review/correct them:

```python
# See Option A above and edit the generated manifest
```

## Step 4: Update Dataloader to Use Manifest (Optional)

If you created a manifest file, you can update the dataloader:

```python
# In src/dataloader/ocr_dataloader.py, modify _load_handwritten_notes:

def _load_handwritten_notes(self) -> List[Tuple[str, str]]:
    """Load handwritten notes from manifest file."""
    manifest_path = os.path.join(self.root_dir, "handwritten_notes_manifest.txt")
    data = []
    
    if not os.path.exists(manifest_path):
        logger.debug(f"Manifest not found: {manifest_path}")
        return []
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                
                img_path, text = parts
                text = clean_text(text)
                
                if text and os.path.exists(img_path):
                    data.append((img_path, text))
    
    except Exception as e:
        logger.warning(f"Error loading handwritten notes manifest: {e}")
    
    logger.debug(f"Handwritten notes loaded: {len(data)} samples")
    return data
```

## Step 5: Train with New Datasets

### Basic Training

```bash
cd /path/to/SmartNotes
python src/training/train_ocr.py
```

The training script will automatically:
1. Load IAM, CensusHWR, and GNHK datasets
2. Load handwritten notes (if extracted)
3. Load printed notes (if extracted)
4. Combine all datasets
5. Split into train/val sets
6. Train the model

### Monitor Training

```bash
# Watch logs
tail -f smartnotes.log

# Check progress
ps aux | grep train_ocr
```

## Step 6: Verify New Samples Were Loaded

Check the logs to verify datasets were loaded:

```bash
grep "loaded:" smartnotes.log
```

Expected output:
```
DEBUG:SmartNotes.dataloader:IAM loaded: 6482 samples
DEBUG:SmartNotes.dataloader:CensusHWR loaded: 3500 samples
DEBUG:SmartNotes.dataloader:GNHK loaded: 1200 samples
DEBUG:SmartNotes.dataloader:Handwritten notes loaded: 450 samples
DEBUG:SmartNotes.dataloader:Printed notes loaded: 380 samples
TRAIN set: 10000 samples loaded
VAL set: 2012 samples loaded
```

## Troubleshooting

### Issue: "pdf2image not installed"

**Solution:**
```bash
pip install pdf2image

# Also install poppler:
# macOS: brew install poppler
# Linux: sudo apt-get install poppler-utils
# Windows: choco install poppler (or download from GitHub)
```

### Issue: "No images extracted"

**Causes:**
- PDFs are corrupted or not standard format
- Missing write permissions to output directory
- poppler not properly installed

**Solutions:**
```bash
# Check if PDFs are readable
file "datasets/handwritten notes"/*.pdf

# Verify output directory exists and is writable
mkdir -p datasets/handwritten_notes_extracted
chmod 755 datasets/handwritten_notes_extracted

# Try extracting a single PDF first
python -c "
from pdf2image import convert_from_path
images = convert_from_path('datasets/handwritten notes/ada/ada.pdf', dpi=150)
print(f'Successfully extracted {len(images)} pages')
"
```

### Issue: "Handwritten notes loaded: 0 samples"

**Causes:**
- PDFs haven't been extracted yet
- Extracted images are in wrong directory
- Image files are corrupted

**Solution:**
```bash
# Verify extracted images exist
find datasets/handwritten_notes_extracted -type f -name "*.png" | head -5

# Check image file sizes
ls -lh datasets/handwritten_notes_extracted/*/

# Try re-extracting with different DPI
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes" \
  --output datasets/handwritten_notes_extracted \
  --dpi 200
```

### Issue: Memory error or slow extraction

**Solution - Extract by folder:**
```bash
# Extract one subject at a time
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes/ada" \
  --output datasets/handwritten_notes_extracted \
  --dpi 150

# Then extract next subject
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes/dsa" \
  --output datasets/handwritten_notes_extracted \
  --dpi 150
```

## Advanced Usage

### Using PDF Processor in Python Code

```python
from src.dataloader.pdf_processor import PDFProcessor, process_notes_directory

# Process an entire directory
total = process_notes_directory(
    notes_dir="datasets/handwritten notes",
    output_dir="datasets/handwritten_notes_extracted",
    dpi=150,
    extract_regions=False  # Set to True to extract text regions
)
print(f"Extracted {total} images")

# Or process a single PDF
processor = PDFProcessor(dpi=200)
images = processor.pdf_to_images("datasets/handwritten notes/ada/ada.pdf")
print(f"Got {len(images)} pages")

# Extract text regions (experimental)
for img in images:
    regions = processor.extract_text_regions(img)
    print(f"Found {len(regions)} potential text regions")
```

### Customizing PDF Extraction Parameters

```python
from src.dataloader.pdf_processor import PDFProcessor

# Higher DPI for better quality (slower, more storage)
processor = PDFProcessor(dpi=300, max_size=2000)

# Lower DPI for faster extraction (lower quality)
processor = PDFProcessor(dpi=100, max_size=1200)

# Process specific page range
images = processor.pdf_to_images(
    "datasets/handwritten notes/ada/ada.pdf",
    start_page=1,      # Start from page 1
    end_page=50        # End at page 50
)
```

## Performance Considerations

### DPI Settings

| DPI | Quality | Speed | Disk Space | Use Case |
|-----|---------|-------|-----------|----------|
| 100 | Low     | Fast  | Small     | Testing, quick iterations |
| 150 | Medium  | Medium| Medium    | **Recommended** |
| 200 | High    | Slow  | Large     | Final training |
| 300 | Very High| Very Slow | Very Large | Archival, detailed text |

### Recommended Workflow

1. **First extraction:** Use DPI 150 (default)
2. **If accuracy is low:** Re-extract with DPI 200
3. **If disk space is an issue:** Use DPI 100 or `extract_regions=True`

### Combining Datasets

The SmartNotes dataloader automatically combines all available datasets:

```
Total training samples = IAM + CensusHWR + GNHK + Handwritten + Printed

Expected sizes:
- IAM: ~6,500 samples
- CensusHWR: ~3,500 samples
- GNHK: ~1,200 samples
- Handwritten notes: 100-1000 samples (depending on PDFs)
- Printed notes: 100-1000 samples (depending on PDFs)

Total: ~13,000+ samples available for training
```

## Next Steps

1. ✅ Extract PDFs: `python -m src.dataloader.pdf_processor ...`
2. ✅ Verify extraction: `find datasets/handwritten_notes_extracted -type f | wc -l`
3. ✅ Train model: `python src/training/train_ocr.py`
4. ✅ Monitor training: `tail -f smartnotes.log`
5. ✅ Evaluate on new data: `python src/inference/test_ocr.py --mode val`

## Support

For issues with PDF extraction:
- Check that poppler is installed: `which pdftoimage` (macOS/Linux)
- Verify PDF files are readable: `file *.pdf`
- Check disk space: `df -h`

For training issues, see TROUBLESHOOTING.md or training logs: `smartnotes.log`
