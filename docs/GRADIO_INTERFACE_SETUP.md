# SmartNotes Gradio Interface - Complete Setup

## 🎯 What You Get

A web-based interface that:
1. **Upload Notes** - Drag & drop handwritten or printed images
2. **Extract Text** - Deep learning OCR converts image to text
3. **Classify Subject** - Automatically identifies course/subject
4. **Show Results** - Displays extracted text, subject, and confidence score

## 🚀 Launch

### Quick Start (One Command)

```bash
cd /Users/kanike/Desktop/SmartNotes/SmartNotes
python scripts/launch_gradio.py
```

Then open: **http://localhost:7860**

### Alternative (Direct)

```bash
python src/inference/demo_gradio_notes.py
```

## 📚 Supported Subjects

The interface can classify these VTU CSE courses:

- **BCS501** - Software Engineering & Project Management
- **BCS502** - Computer Networks  
- **BCS503** - Theory of Computation
- **BCSL504** - Web Technology Lab
- **BCS515A** - Computer Graphics
- **BCS515B** - Artificial Intelligence
- **BCS515C** - Unix System Programming
- **BCS515D** - Network Security
- **BCS515E** - Compiler Design
- **BCS515F** - Internet of Things
- **BCS515G** - Python for Data Analytics
- And more...

## 🔧 How It Works

### Architecture

```
User Uploads Image
        ↓
    [Gradio Interface]
        ↓
   [OCRRecognizer]  ← CRNN + BiLSTM model
        ↓
  Extract Text
        ↓
[Subject Classifier] ← Keyword-based matching
        ↓
 Classify Subject
        ↓
Display: Text | Subject | Confidence | Keywords
```

### Technical Stack

- **OCR Model**: CRNN (Convolutional Recurrent Neural Network)
  - CNN: Feature extraction (1→512 channels)
  - BiLSTM: Sequence modeling (256 hidden units)
  - CTC Loss: Sequence alignment

- **Classification**: Keyword-based matching
  - VTU subject keywords database
  - Confidence: % of matched keywords
  - Fallback: "Unknown Subject" if < 10% match

- **Interface**: Gradio
  - Zero code needed to use
  - Real-time processing
  - Beautiful UI

## 📁 Files Created

```
SmartNotes/
├── src/inference/
│   └── demo_gradio_notes.py    ← Main Gradio app
└── scripts/
    ├── launch_gradio.py         ← Easy launcher
    └── GRADIO_INTERFACE_GUIDE.md ← Full documentation
```

## ⚙️ Configuration

### Change OCR Model

Edit `src/inference/demo_gradio_notes.py`:

```python
self.recognizer = OCRRecognizer(
    checkpoint_path="checkpoints/ocr_best.pth"  # Change model here
)
```

### Add Custom Keywords

Edit `preprocessing/subject_classifier.py`:

```python
VTU_SUBJECT_KEYWORDS = {
    "Your Subject": [
        "keyword1", "keyword2", "keyword3", ...
    ]
}
```

### Customize Interface

Edit `src/inference/demo_gradio_notes.py`:

- Change title, description, theme
- Add example images
- Modify output formats
- Add more processing steps

## 🌐 Deployment Options

### 1. Local (Default)
```bash
python scripts/launch_gradio.py
# Access: http://localhost:7860
```

### 2. Public Link (Share Mode)
```python
interface.launch(share=True)  # In demo_gradio_notes.py
# Generates temporary public URL
```

### 3. Docker
```bash
docker build -t smartnotes .
docker run -p 7860:7860 smartnotes
```

### 4. Cloud (Hugging Face Spaces, Gradio Cloud)
- Upload files to GitHub
- Connect to Hugging Face Spaces
- Auto-deploys with live link

## 📊 Example Usage

### Workflow

1. **Start Interface**
   ```bash
   python scripts/launch_gradio.py
   ```

2. **Upload Image**
   - Click "Upload Notes"
   - Select PNG/JPG/GIF
   - Or drag & drop

3. **Get Results**
   ```
   Extracted Text: "Software engineering is..."
   Predicted Subject: BCS501 - Software Engineering & Project Management
   Confidence: 0.92 (92%)
   Keywords: software, engineering, project, management, ...
   ```

4. **Copy/Share Results**
   - Copy text to clipboard
   - Share subject classification
   - Export as JSON/PDF (optional)

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Download `checkpoints/ocr_finetuned_stage2_best.pth` |
| "No text detected" | Use higher resolution images (300+ DPI) |
| "Wrong subject" | Add more keywords to `subject_classifier.py` |
| "Port 7860 in use" | Change port: `server_port=7861` in code |
| "Can't import gradio" | `pip install gradio` |

## 📈 Performance

- **OCR Speed**: ~2-5 seconds per page (depends on resolution)
- **Subject Classification**: < 1 second
- **GPU Support**: Automatic (MPS for Apple Silicon)
- **Memory**: ~500MB (model + processing)

## 🎓 VTU Subjects Supported

Based on CSE 5th Semester (Scheme 2022):

**Core Courses:**
- BCS501: Software Engineering & Project Management
- BCS502: Computer Networks
- BCS503: Theory of Computation
- BCSL504: Web Technology Lab

**Professional Electives:**
- BCS515A: Computer Graphics
- BCS515B: Artificial Intelligence
- BCS515C: Unix System Programming
- BCS515D: Network Security
- BCS515E: Compiler Design
- BCS515F: Internet of Things
- BCS515G: Python for Data Analytics

**Others:** Add to `preprocessing/subject_classifier.py`

## 📚 Next Steps

1. **Train Your Model** (improve OCR)
   ```bash
   python src/training/train_ocr.py
   ```

2. **Fine-tune for Handwriting**
   - Use `scripts/setup_notes_integration.py` to add training data
   - Run: `python src/training/finetune_ocr.py`

3. **Add More Features**
   - PDF processing
   - Batch uploads
   - Result export
   - Email integration

## 🤝 Support

For issues:
1. Check `GRADIO_INTERFACE_GUIDE.md` in scripts/
2. Review preprocessing/recognize.py
3. Check subject_classifier.py keywords

---

**Enjoy SmartNotes! 📖✨**
