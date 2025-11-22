# SmartNotes System Architecture

## Overview

SmartNotes is an intelligent academic note organization system that automates the classification and organization of handwritten and printed notes aligned with the Visvesvaraya Technological University (VTU) curriculum. The system implements a resource-optimized multimodal deep learning pipeline with semantic and visual processing streams.

## System Components

### 1. **Semantic Stream** (OCR + Text Analysis)

The semantic stream handles text extraction and understanding:

```
PDF Input 
  ↓
[Page Extraction] → Convert PDF pages to images (DPI: 200)
  ↓
[OCR Engine] → CRNN (Convolutional Recurrent Neural Network)
  │
  ├─ CNN Backbone (5 layers)
  │  - Feature extraction from grayscale images
  │  - Input: 128×32 normalized grayscale
  │  - Output: 512-dim features per position
  │
  ├─ BiLSTM Layers (2 layers, 256 units per direction)
  │  - Sequence modeling of extracted features
  │  - Captures long-range dependencies in text
  │  - Output: 512-dim contextual vectors
  │
  └─ CTC Decoder
     - Handles variable-length sequences
     - No explicit character-level alignment needed
  ↓
[Language Model Enhancement] (Optional)
  - 4-gram ARPA language model
  - KenLM integration for probability scoring
  - Rescores OCR predictions for improved accuracy
  ↓
[Extracted Text]
  ↓
[Subject Classification] → Keyword-based matching
  - VTU curriculum keywords indexed per subject
  - Subject codes: BCS501, BCS502, BCS503, etc.
  - Confidence scoring based on keyword matches
  ↓
[Classified Subject + Confidence]
```

**Key Files:**
- `src/model/ocr_model.py` - CRNN architecture definition
- `src/inference/recognize.py` - OCRLMInference class (330 lines)
- `preprocessing/subject_classifier.py` - Subject classification (keyword-based)
- `lm/smartnotes.arpa` - 4-gram language model (767KB)

**Performance Metrics:**
- Character Error Rate (CER): 4.65% ± 11.68%
- Perfect Recognition: 75.78% (3,789/5,000 samples)
- Excellent Quality (CER ≤ 15%): 91.08%

### 2. **Visual Stream** (Layout Analysis)

[Future Implementation - Currently Placeholder]

The visual stream processes layout and structural features:

```
Image Input
  ↓
[CNN Feature Extractor]
  - Lightweight CNN (3-4 layers)
  - Captures layout-based features
  - Detects text blocks, spacing, margins
  - Output: Layout embeddings
  ↓
[Layout Features]
```

**Design Intent:**
- Capture document structure independently of text content
- Provide complementary signal for subject classification
- Support on-device inference (TFLite quantization)

### 3. **Fusion Mechanism** (Late Fusion)

Combines semantic and visual streams:

```
Semantic Embeddings          Visual Embeddings
      ↓                              ↓
      └──────────→ [Fusion Layer] ←──────────
                         ↓
                  [Late Fusion MLP]
                    - 2-3 hidden layers
                    - Learnable fusion weights
                         ↓
                  [Subject Prediction]
                    - VTU subject code output
```

**Current Implementation:** Semantic stream only (visual placeholder)
**Phase 2:** Full fusion mechanism with visual CNN

### 4. **File Organization** (Subject-based Grouping)

After classification, files are organized by subject:

```
Output Directory/
├── BCS501/
│   ├── page_0001.png          [Extracted page image]
│   ├── page_0002.png
│   ├── metadata.json          [Subject-specific metadata]
│   └── recognized_text.txt    [OCR output for all pages]
├── BCS502/
│   ├── page_0001.png
│   └── ...
├── BCS503/
│   └── ...
├── processing_metadata.json   [Global processing log]
└── summary_report.json        [Batch statistics]
```

## Data Pipeline

### Input: PDF Documents

- **Format:** Standard PDF (scanned or digital)
- **Pages:** Variable length documents
- **Content:** Handwritten or printed academic notes
- **Preprocessing:** Auto-rotated, auto-scaled to 128×32

### Processing Steps

```mermaid
1. PDF Input
   ↓
2. Page Extraction (pdf2image)
   - Convert PDF pages to PIL Images
   - DPI: 200 (tunable)
   ↓
3. OCR (CRNN-EPOCH6)
   - Load model: checkpoints/ocr_epoch_6.pth (73MB)
   - Resize to 128×32
   - Normalize to [0,1]
   - CTC greedy decoding
   ↓
4. LM Enhancement (Optional)
   - Load ARPA model: lm/smartnotes.arpa (767KB)
   - Rescore predictions with probability
   - Blend with OCR confidence
   ↓
5. Subject Classification
   - Extract keywords from recognized text
   - Match against VTU subject keywords
   - Assign subject code + confidence score
   ↓
6. File Organization
   - Create subject directories
   - Save page images + metadata
   - Generate processing reports
   ↓
7. Output: Organized Folder Structure
```

## Technology Stack

### Deep Learning
- **PyTorch 2.x** - Model training and inference
- **CRNN Architecture** - Sequence-to-sequence text recognition
- **CTC Loss** - Handles variable-length sequences

### Language Models
- **KenLM** - Fast n-gram language model loading
- **ARPA Format** - Standard LM file format (4-gram model)

### PDF Processing
- **pdf2image** - Convert PDF pages to images
- **Pillow (PIL)** - Image manipulation
- **OpenCV** - Image resizing and normalization

### Deployment
- **Python 3.12.3** - Runtime environment
- **Argparse** - CLI argument parsing
- **JSON** - Metadata serialization

### Hardware Support
- **CPU** - Full support
- **CUDA** - NVIDIA GPUs (auto-detection)
- **MPS** - Apple Silicon (native acceleration)

## API Reference

### 1. **OCRLMInference Class**

Main inference engine combining OCR + Language Model:

```python
from src.inference.recognize import OCRLMInference

# Initialize
ocr = OCRLMInference(use_lm=True, device="auto")

# Single image inference
result = ocr.infer_single("path/to/image.png")
# Returns: {"prediction": "text", "device": "...", ...}

# Batch inference
batch_results = ocr.infer(image_tensor, label_tensor)
# Returns: List[(pred_text, gt_text, cer, wer), ...]

# Evaluate on dataset
avg_cer, avg_wer = ocr.evaluate_on_dataset(mode="val", num_samples=5000)
```

### 2. **PDFProcessor Class**

End-to-end PDF processing pipeline:

```python
from src.inference.pdf_processor import PDFProcessor

# Initialize
processor = PDFProcessor(use_lm=True, device="auto")

# Process single PDF
result = processor.process_pdf("path/to/document.pdf", "output_dir/")

# Process batch
results = processor.process_batch(
    "pdf_directory/",
    "output_dir/",
    pattern="*.pdf",
    organize_by_subject=True
)

# Generate report
processor.generate_summary_report(results, "summary.json")
```

### 3. **SmartNotes CLI**

End-to-end command-line interface:

```bash
# Single PDF processing
python3 smartnotes_cli.py --pdf document.pdf --output results/

# Batch processing
python3 smartnotes_cli.py --batch ./pdfs --output results/ --organize

# With options
python3 smartnotes_cli.py \
  --batch ./pdfs \
  --output results/ \
  --use-lm \
  --device mps \
  --organize \
  --html \
  --verbose
```

### 4. **Subject Classifier**

Keyword-based subject classification:

```python
from preprocessing.subject_classifier import classify_subject

text = "Explain TCP three-way handshake and congestion control"
subject, keywords, confidence = classify_subject(text)
# subject: "BCS502 - Computer Networks"
# keywords: ["tcp", "three-way handshake", "congestion control"]
# confidence: 0.87
```

## Configuration

All configuration centralized in `config.py`:

```python
class InferenceConfig:
    CHECKPOINT_PATH = "ocr_epoch_6.pth"    # OCR model checkpoint
    LM_PATH = "lm/smartnotes.arpa"         # Language model path
    LM_WEIGHT = 0.3                        # LM blending weight
    BEAM_WIDTH = 5                         # Reserved for beam search
    USE_CPU = False                        # Force CPU usage
    
class PreprocessingConfig:
    USE_SUBJECT_CLASSIFIER = True          # Enable classification
    LINE_SEGMENT_METHOD = "adaptive"       # Line segmentation method
    CLEAN_TEXT = True                      # Text cleaning
```

## Performance Characteristics

### Model Performance
- **CER:** 4.65% ± 11.68% (on 5000 validation samples)
- **Perfect Recognition:** 75.78%
- **Excellent Quality:** 91.08% (CER ≤ 15%)

### Processing Speed
- **OCR per page:** ~500ms (CPU), ~100ms (GPU)
- **Subject classification:** <10ms per page
- **Batch processing:** 50-100 pages/minute (CPU)

### Memory Requirements
- **OCR Model:** 73MB (checkpoint)
- **LM Model:** 767KB (ARPA format)
- **Runtime:** 2-4GB (CPU), 4-8GB (GPU)

## Design Patterns

### 1. **Resource Optimization**
- Lightweight CRNN (73MB) vs. 200MB+ alternatives
- On-device inference capability (TFLite quantization ready)
- Efficient batch processing with streaming

### 2. **Modularity**
- Separate semantic and visual streams (future)
- Pluggable LM support (KenLM or disabled)
- Extensible subject classifier (keyword-based)

### 3. **Robustness**
- Cross-platform support (CPU/CUDA/MPS)
- Graceful degradation (works without LM)
- Comprehensive error handling and logging

### 4. **Maintainability**
- Type hints throughout
- Extensive docstrings
- JSON-based configuration
- Unit tests in `tests/`

## Future Enhancements

### Phase 2: Visual Stream
- Implement CNN-based layout feature extractor
- Train fusion MLP on semantic+visual features
- Achieve improved accuracy through multimodal fusion

### Phase 3: Web Interface
- Flask/FastAPI REST API
- Web dashboard for batch processing
- Real-time progress tracking

### Phase 4: Mobile Deployment
- TFLite quantization (INT8)
- Flutter-based mobile app
- On-device inference (2-3MB model)

### Phase 5: Advanced Features
- Beam search decoding with LM integration
- Multi-language support
- Custom subject taxonomy

## File Structure

```
SmartNotes/
├── src/
│   ├── model/              # Model definitions
│   │   └── ocr_model.py    # CRNN architecture
│   ├── training/           # Training scripts
│   ├── inference/          # Inference pipeline
│   │   ├── recognize.py    # OCRLMInference class
│   │   ├── pdf_processor.py  # PDF processing pipeline
│   │   └── cli_recognize.py  # Image CLI tool
│   └── dataloader/         # Dataset utilities
├── preprocessing/          # Preprocessing pipeline
│   ├── subject_classifier.py  # VTU subject classification
│   └── pipeline.py         # End-to-end pipeline
├── checkpoints/            # Model weights
│   └── ocr_epoch_6.pth     # Trained CRNN (v1.0)
├── lm/                     # Language models
│   └── smartnotes.arpa     # 4-gram ARPA LM
├── config.py               # Centralized configuration
├── smartnotes_cli.py       # Main CLI application
└── README.md               # Full documentation
```

## Conclusion

SmartNotes implements a production-ready OCR system specifically designed for academic note organization. The semantic stream achieves 4.65% CER on validated data, with the framework ready for visual stream integration and advanced features in future phases. The system balances performance, efficiency, and maintainability for practical deployment.
