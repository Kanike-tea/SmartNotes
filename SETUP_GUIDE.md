# SmartNotes Setup & Diagnostic Guide

## Quick Start

### 1. System Check
```bash
python system_check.py
```
**What it does:** Validates Python version, dependencies, models, datasets, and source code

**Expected output:**
- ✓ Python 3.8+
- ✓ PyTorch, OpenCV, NumPy, Gradio
- ✓ Model checkpoints (ocr_best.pth, ocr_final.pth)
- ✓ Dataset directories
- ✓ All source modules

**What to do if it fails:**
- Missing Python packages → Run `pip install -r requirements.txt`
- Missing model files → Download from `/checkpoints/`
- Missing datasets → Check `/datasets/` directory structure

---

### 2. Quick OCR Test
```bash
python quick_test.py path/to/image.png
```
**What it does:** 4-step end-to-end test on a single image

**Steps:**
1. Load image
2. Load model
3. Preprocess (line segmentation)
4. Recognize text

**Expected output:**
- Image dimensions
- Number of lines detected
- Recognized text
- Total characters

**What to do if it fails:**
- Image format issue → Use PNG/JPG
- Model not loaded → Check `system_check.py` first
- Line detection failed → Check image quality (minimum 100x100 pixels)

---

## Diagnostic Workflow

### When OCR Returns [NO TEXT DETECTED]
1. **Check image quality:**
   ```bash
   python quick_test.py your_image.png
   ```
   - Verify image loads correctly
   - Check line count
   - Look for error messages in preprocessing

2. **Run full diagnostic:**
   ```bash
   python diagnostics.py --image your_image.png --verbose
   ```
   - Get detailed preprocessing steps
   - See line segment coordinates
   - Check confidence scores

3. **Review preprocessing:**
   - Image should have good contrast
   - Text should be clearly visible
   - Lines should be horizontal (script expects English text)

### When Model Fails to Load
1. **System check:**
   ```bash
   python system_check.py
   ```
   - Verify PyTorch is installed
   - Check model files exist

2. **Check CUDA (if using GPU):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Force CPU mode:**
   - Edit `src/model/ocr_model.py`
   - Change device to 'cpu'

### When Dependencies Are Missing
1. **Reinstall requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For GPU support (if available):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check specific package:**
   ```bash
   python -c "import [package_name]; print([package_name].__version__)"
   ```

---

## File Structure Reference

```
SmartNotes/
├── system_check.py           # Validates setup
├── quick_test.py             # Tests OCR on one image
├── diagnostics.py            # Detailed troubleshooting
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── model/
│   │   └── ocr_model.py      # Main OCR model
│   ├── training/
│   │   ├── train_ocr.py      # Training script
│   │   └── finetune_ocr.py   # Fine-tuning script
│   └── inference/
│       ├── test_ocr.py       # Test inference
│       └── demo_gradio.py    # Gradio web interface
│
├── preprocessing/
│   ├── recognize.py          # Recognition pipeline
│   ├── line_segment.py       # Line segmentation
│   ├── postprocess.py        # Post-processing
│   └── pipeline.py           # Full pipeline
│
├── checkpoints/
│   ├── ocr_best.pth          # Best model
│   └── ocr_final.pth         # Final model
│
└── datasets/
    ├── GNHK/                 # Ground truth handwriting
    ├── CensusHWR/            # Census handwriting
    └── IAM/                  # IAM database
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size in training config
- Use CPU device instead
- Reduce image resolution

### Issue: "Image not found" in quick_test.py
**Solution:**
- Use absolute path or relative path from SmartNotes directory
- Check file permissions
- Verify file format (PNG/JPG)

### Issue: "No lines detected" in preprocessing
**Solution:**
- Check image contrast
- Ensure image is not rotated
- Minimum image size: 100x100 pixels
- Check for:
  - Poor lighting
  - Very thin text
  - Skewed handwriting

### Issue: "Empty recognition result"
**Solution:**
- Check model checkpoint exists and loads
- Verify image format
- Try `diagnostics.py --verbose` to see intermediate outputs
- Check line segmentation in preprocessing

---

## Testing Workflow

### Step 1: Validate Environment
```bash
python system_check.py
```

### Step 2: Test Single Image
```bash
python quick_test.py datasets/handwritten\ notes/ada/page1.png
```

### Step 3: Detailed Diagnostics
```bash
python diagnostics.py --image datasets/handwritten\ notes/ada/page1.png --verbose
```

### Step 4: Run Full Pipeline
```bash
python src/inference/test_ocr.py --image test_image.png
```

### Step 5: Launch Gradio Interface
```bash
python src/inference/demo_gradio.py
```

---

## Performance Tips

- **Faster processing:** Use CPU if GPU VRAM is limited
- **Better accuracy:** Use `ocr_best.pth` instead of `ocr_final.pth`
- **Batch processing:** Use pipeline scripts instead of single image
- **Memory efficient:** Process smaller batches in training

---

## Getting Help

When reporting issues, include:
1. Output from `system_check.py`
2. Output from `quick_test.py` with your image
3. Python version: `python --version`
4. PyTorch version: `python -c "import torch; print(torch.__version__)"`
5. Image that fails (if possible)

---

## Next Steps

After setup is validated:
- See `TRAINING.md` for training new models
- See `INFERENCE.md` for running inference
- Check `src/inference/demo_gradio.py` for web interface
- Review `preprocessing/pipeline.py` for custom pipelines
