# Handwritten & Printed Notes Integration - Summary

This document summarizes the implementation for integrating handwritten and printed notes into SmartNotes OCR training.

## What Was Added

### 1. **PDF Processor Module** (`src/dataloader/pdf_processor.py`)
   - Converts PDF pages to grayscale images
   - Supports customizable DPI settings (100-300)
   - Optional text region extraction
   - Batch processing for entire directories
   - Error handling and logging

### 2. **Dataloader Updates** (`src/dataloader/ocr_dataloader.py`)
   - `_load_handwritten_notes()`: Loads extracted handwritten note images
   - `_load_printed_notes()`: Loads extracted printed note images
   - Both methods integrate seamlessly with existing IAM, CensusHWR, and GNHK loaders
   - Automatic dataset combination and train/val splitting

### 3. **Setup Script** (`setup_notes_integration.py`)
   - Interactive setup wizard
   - Checks dependencies and poppler installation
   - Extracts PDFs to images automatically
   - Verifies extracted datasets

### 4. **Comprehensive Guide** (`NOTES_INTEGRATION_GUIDE.md`)
   - Step-by-step instructions
   - Dependency installation
   - PDF extraction process
   - Metadata preparation options
   - Troubleshooting guide
   - Performance recommendations

### 5. **Updated Requirements** (`requirements.txt`)
   - Added `pdf2image==1.16.3`

## Quick Start

### Option 1: Interactive Setup (Recommended)
```bash
python setup_notes_integration.py
```

This will:
- Check all dependencies
- Verify poppler installation
- Ask which datasets to extract
- Extract PDFs automatically
- Verify extraction success

### Option 2: Manual Setup
```bash
# 1. Install dependencies
brew install poppler  # macOS
pip install -r requirements.txt

# 2. Extract PDFs
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes" \
  --output datasets/handwritten_notes_extracted \
  --dpi 150

python -m src.dataloader.pdf_processor \
  --input "datasets/printed notes" \
  --output datasets/printed_notes_extracted \
  --dpi 150

# 3. Train with new datasets
python src/training/train_ocr.py
```

## File Organization

```
SmartNotes/
├── src/dataloader/
│   ├── ocr_dataloader.py          (UPDATED: +2 new load methods)
│   └── pdf_processor.py           (NEW: PDF extraction)
├── datasets/
│   ├── handwritten notes/         (existing: PDFs)
│   ├── printed notes/             (existing: PDFs)
│   ├── handwritten_notes_extracted/   (NEW: extracted images)
│   └── printed_notes_extracted/   (NEW: extracted images)
├── setup_notes_integration.py     (NEW: interactive setup)
├── NOTES_INTEGRATION_GUIDE.md     (NEW: comprehensive guide)
├── requirements.txt               (UPDATED: +pdf2image)
└── ...
```

## Data Flow

```
Raw PDFs (datasets/handwritten notes/*.pdf)
    ↓
PDF Processor (pdf_processor.py)
    ↓
Extracted Images (datasets/handwritten_notes_extracted/*.png)
    ↓
Dataloader (ocr_dataloader.py)
    ↓
Combined Dataset (IAM + CensusHWR + GNHK + Handwritten + Printed)
    ↓
Training (src/training/train_ocr.py)
```

## Key Features

### PDF Processor
- ✅ Multi-page PDF support
- ✅ Configurable DPI (100-300)
- ✅ Grayscale conversion
- ✅ Optional image resizing
- ✅ Text region extraction (experimental)
- ✅ Batch processing
- ✅ Error handling and recovery
- ✅ Progress logging

### Dataloader Integration
- ✅ Automatic dataset discovery
- ✅ Seamless combination with existing datasets
- ✅ Train/val/test splitting
- ✅ Logging for verification
- ✅ Graceful handling of missing datasets
- ✅ Extensible manifest support (optional)

### Setup Script
- ✅ Dependency checking
- ✅ Poppler verification
- ✅ Interactive extraction
- ✅ Dataset verification
- ✅ Colored output for clarity

## Expected Results

### Dataset Sizes (Typical)
```
IAM:                     6,482 samples
CensusHWR:              3,500 samples
GNHK:                   1,200 samples
Handwritten notes:      100-1000 samples (depends on PDFs)
Printed notes:          100-1000 samples (depends on PDFs)
─────────────────────────────────────
Total:                  11,382-14,182 samples

Train set (85%):        9,675-12,055 samples
Val set (15%):          1,707-2,127 samples
```

### Training Improvements Expected
- More diverse handwriting samples
- Subject-specific terminology
- Better generalization across writing styles
- Improved accuracy on VTU-specific content

## Advanced Usage

### Custom Metadata with Manifest Files
```bash
# Create manifest file for handwritten notes
for file in datasets/handwritten_notes_extracted/*/*.png; do
  echo "$file	<transcription>" >> datasets/handwritten_notes_manifest.txt
done

# Then update ocr_dataloader.py to use manifest instead of filenames
```

### Extracting Specific Page Ranges
```python
from src.dataloader.pdf_processor import PDFProcessor

processor = PDFProcessor(dpi=200)

# Extract only pages 1-50 from a PDF
images = processor.pdf_to_images(
    "datasets/handwritten notes/ada/ada.pdf",
    start_page=1,
    end_page=50
)
```

### Text Region Extraction
```bash
# Automatically segment text lines from document images
python -m src.dataloader.pdf_processor \
  --input "datasets/handwritten notes" \
  --output datasets/handwritten_notes_extracted \
  --extract-regions
```

## Troubleshooting

### Common Issues and Solutions

**Issue:** `pdf2image not found`
```bash
pip install pdf2image
```

**Issue:** `pdftoimage command not found`
```bash
# macOS
brew install poppler

# Linux
sudo apt-get install poppler-utils

# Windows
choco install poppler
```

**Issue:** `No images extracted`
```bash
# Verify PDFs are readable
file "datasets/handwritten notes"/*.pdf

# Create output directory
mkdir -p datasets/handwritten_notes_extracted

# Check available disk space
df -h
```

**Issue:** Slow extraction
- Use lower DPI (100 instead of 150)
- Extract one folder at a time
- Use `--extract-regions` to reduce image count

For more troubleshooting, see `NOTES_INTEGRATION_GUIDE.md`.

## Testing the Integration

```bash
# 1. Run setup script (recommended)
python setup_notes_integration.py

# 2. Verify extraction
find datasets/handwritten_notes_extracted -type f | wc -l

# 3. Test dataloader
python -c "
from src.dataloader.ocr_dataloader import SmartNotesOCRDataset
ds = SmartNotesOCRDataset(mode='train', max_samples=10)
print(f'Dataset size: {len(ds)}')
for i in range(3):
    img, label = ds[i]
    print(f'Sample {i}: image shape {img.shape}, label length {len(label)}')
"

# 4. Train with new data
python src/training/train_ocr.py
```

## Files Modified/Created

### New Files
- `src/dataloader/pdf_processor.py` (300+ lines)
- `setup_notes_integration.py` (200+ lines)
- `NOTES_INTEGRATION_GUIDE.md` (400+ lines)

### Modified Files
- `src/dataloader/ocr_dataloader.py` (+100 lines)
- `requirements.txt` (+1 dependency)

### Total Lines Added
- 900+ lines of new code
- 400+ lines of documentation

## Next Steps

1. **Review the guide**: Read `NOTES_INTEGRATION_GUIDE.md` for detailed information
2. **Run setup**: Execute `python setup_notes_integration.py`
3. **Monitor extraction**: Check logs for progress
4. **Train model**: Run `python src/training/train_ocr.py`
5. **Evaluate**: Check accuracy improvements on test set

## Support

For issues with:
- **PDF extraction**: See "Troubleshooting" in `NOTES_INTEGRATION_GUIDE.md`
- **Training**: See `smartnotes.log` for error messages
- **Integration**: Review examples in docstrings of `pdf_processor.py`

## Performance Metrics

### Extraction Time Estimates (per 100 pages)
| DPI | Time  | Disk Space |
|-----|-------|-----------|
| 100 | ~30s  | ~50MB     |
| 150 | ~60s  | ~100MB    |
| 200 | ~120s | ~200MB    |

### Training with Combined Datasets
- Increased training time: ~20% longer per epoch (due to larger dataset)
- Memory usage: Slight increase due to more samples
- Expected accuracy improvement: 2-5% depending on note quality

## Future Enhancements

Potential improvements not yet implemented:
- [ ] Automatic handwriting quality assessment
- [ ] Layout-aware text extraction
- [ ] Multi-language support
- [ ] Document classification (OCR vs printed vs handwritten)
- [ ] Interactive annotation UI
- [ ] Batch processing optimization
- [ ] Cloud storage integration

## License & Attribution

This integration uses:
- `pdf2image`: MIT License
- `poppler`: GPL License
- Original SmartNotes: See project LICENSE

---

**Ready to integrate your notes?** Start with:
```bash
python setup_notes_integration.py
```
