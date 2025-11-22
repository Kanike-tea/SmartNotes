# START HERE - SmartNotes v2.0 Usage Guide

## ⚡ Get Started in 60 Seconds

### Step 1: Navigate to Project

### Step 2: Process Your First PDF
```bash
python3 smartnotes_cli.py --pdf my_notes.pdf --output results/
```

### Step 3: Check Output
```bash
open results/  # macOS
# or
ls results/    # Linux/Windows
```

Your notes are now organized by VTU subject! 🎉

---

## 📖 What You Just Did

1. **PDF was extracted** into individual pages
2. **Each page was recognized** using OCR (4.65% accuracy)
3. **Text was classified** into VTU subjects (BCS501, BCS502, etc.)
4. **Files were organized** into subject folders automatically
5. **Metadata was saved** with confidence scores

---

## 🎯 Common Commands

### Batch Process Multiple PDFs
```bash
python3 smartnotes_cli.py --batch ./my_pdfs --output results/ --organize
```

### Generate HTML Report
```bash
python3 smartnotes_cli.py --batch ./my_pdfs --output results/ --html
```

### Use Language Model (slower but more accurate)
```bash
python3 smartnotes_cli.py --pdf notes.pdf --output results/ --use-lm
```

### Disable Language Model (faster)
```bash
python3 smartnotes_cli.py --pdf notes.pdf --output results/ --no-lm
```

### Use GPU (much faster)
```bash
python3 smartnotes_cli.py --batch ./pdfs --output results/ --device cuda
```

---

## 📁 Expected Output

After processing, your results folder will look like:

```
results/
├── BCS501 - Software Engineering/
│   ├── page_0001.png        # Extracted page image
│   ├── page_0002.png
│   └── metadata.json         # Classification metadata
├── BCS502 - Computer Networks/
│   ├── page_0001.png
│   └── ...
├── BCS503 - Theory of Computation/
│   └── ...
├── processing_metadata.json   # Full processing log
├── summary_report.json        # Statistics
└── summary.html              # Visual dashboard (if --html used)
```

---

## 🔍 Understanding the Output

### metadata.json (per subject)
```json
{
  "subject_code": "BCS502",
  "pages": [
    {
      "page_number": 1,
      "source_pdf": "notes.pdf",
      "extracted_text": "TCP three-way handshake...",
      "subject_name": "BCS502 - Computer Networks",
      "confidence": 0.87,
      "keywords_matched": ["tcp", "routing", "ip"]
    }
  ]
}
```

### summary_report.json (overall stats)
```json
{
  "total_pdfs": 5,
  "total_pages": 50,
  "successfully_processed": 48,
  "by_subject": {
    "BCS501": 10,
    "BCS502": 15,
    "BCS503": 23
  },
  "by_confidence": {
    "high": 35,
    "medium": 10,
    "low": 3
  }
}
```

---

## 🐍 Python Integration

Use SmartNotes as a library in your code:

```python
from src.inference.pdf_processor import PDFProcessor

# Initialize
processor = PDFProcessor(use_lm=True, device="auto")

# Process single PDF
result = processor.process_pdf("document.pdf", "output_dir/")
print(f"Processed {result.processed_pages}/{result.total_pages} pages")

# Process batch
results = processor.process_batch(
    "pdf_directory/",
    "output_dir/",
    organize_by_subject=True
)

# Generate report
processor.generate_summary_report(results, "summary.json")
```

---

## 📚 VTU Subjects Supported

The system can classify notes into these subjects:

| Code | Subject |
|------|---------|
| BCS501 | Software Engineering & Project Management |
| BCS502 | Computer Networks |
| BCS503 | Theory of Computation |
| BCSL504 | Web Technology Lab |
| BCS515A | Computer Graphics |
| BCS515B | Artificial Intelligence |
| BCS515C | Unix System Programming |
| BCS515D | Distributed Systems |
| BRMK557 | Research Methodology and IPR |
| BCS508 | Environmental Studies & E-waste |

---

## 🚨 Troubleshooting

### "pdf2image not found"
```bash
pip install pdf2image

# Also install poppler:
brew install poppler  # macOS
# or
sudo apt-get install poppler-utils  # Linux
```

### "Module not found"
```bash
# Make sure you're in the right directory
cd /Users/kanike/Desktop/SmartNotes/SmartNotes

# Or add to path
export PYTHONPATH=$PWD:$PYTHONPATH
```

### "Out of memory"
- Use `--device cpu` instead of GPU
- Process fewer PDFs at once
- Reduce batch size in `config.py`

### "No PDFs found"
```bash
# Check your path
ls -la ./my_pdfs/

# Verify PDF files exist
file ./my_pdfs/*.pdf
```

---

## 🔧 Advanced Options

### All CLI Options
```bash
python3 smartnotes_cli.py --help
```

Shows:
- `--pdf` - Single PDF file
- `--batch` - Batch directory
- `--output` - Output folder
- `--pattern` - File pattern (*.pdf, *.PDF, etc.)
- `--use-lm` / `--no-lm` - Language model toggle
- `--device` - CPU/CUDA/MPS selection
- `--organize` / `--no-organize` - Folder organization
- `--html` - Generate HTML dashboard
- `--verbose` - Detailed logging

---

## 📖 Full Documentation

For more detailed information:

- **README.md** - Comprehensive project documentation
- **SYSTEM_ARCHITECTURE.md** - Technical system design
- **QUICKSTART.md** - Quick reference guide
- **IMPLEMENTATION_SUMMARY.md** - What was built and why
- **DEPLOYMENT_SUMMARY.md** - v1.0 release notes

---

## ✅ Quick Checklist

Before using SmartNotes:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Models present:
  - `checkpoints/ocr_epoch_6.pth` (73MB)
  - `lm/smartnotes.arpa` (767KB)
- [ ] CLI works: `python3 smartnotes_cli.py --help`
- [ ] Test PDF processed successfully

---

## 🎉 You're Ready!

Your SmartNotes system is fully operational and ready to organize your academic notes.

**Start processing:**
```bash
python3 smartnotes_cli.py --batch ./my_pdfs --output ./organized --html
```

**Questions?** Check the documentation files or explore the source code in `src/inference/`.

---

**Version**: 2.0
**Status**: Production Ready
**Last Updated**: 22-11-2025
