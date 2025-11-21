# SmartNotes Quick Start Guide

Get SmartNotes up and running in 5 minutes!

## Installation

### Option 1: Using pip (Recommended)

```bash
# Clone repository
git clone https://github.com/Kanike-tea/SmartNotes.git
cd SmartNotes

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Docker

```bash
docker build -t smartnotes:latest .
docker run -v $(pwd)/datasets:/app/datasets -v $(pwd)/checkpoints:/app/checkpoints smartnotes:latest
```

### Option 3: Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Test

Verify installation:

```bash
python -c "from src.model.ocr_model import CRNN; print('✓ Installation successful')"
```

## Run Inference (30 seconds)

### With Pre-trained Model

```bash
cd src/inference
python test_ocr.py --num-samples 5
```

Expected output:
```
Sample 1
Predicted: hello world
Ground Truth: hello world
CER: 0.0000, WER: 0.0000
```

### Custom Image

```python
from src.inference.test_ocr import OCRInference
import torch

inference = OCRInference(checkpoint_path='checkpoints/ocr_best.pth')
image = torch.randn(1, 1, 32, 128)  # Your image here
prediction = inference.predict(image)
print(f"Predicted: {prediction}")
```

## Training (Requires Dataset)

### 1. Prepare Data

Download and place datasets in `datasets/`:

```
datasets/
├── IAM/              # Download from http://www.fki.inf.unibe.ch/...
├── CensusHWR/       # Or your custom dataset
└── GNHK/            # Optional
```

### 2. Configure Training

Edit `config.py` for your setup:

```python
class TrainingConfig:
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
```

### 3. Start Training

```bash
cd src/training
python train_ocr.py
```

Monitor progress in `smartnotes.log`:

```bash
tail -f smartnotes.log
```

## Common Commands

```bash
# Run tests
pytest tests/ -v

# Check model architecture
python -c "from src.model.ocr_model import CRNN; m = CRNN(36); print(m)"

# View configuration
python -c "from config import Config; Config.print_config()"

# Format code
black src/ preprocessing/

# Lint code
flake8 src/

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

## Using Configuration System

Edit `config.py` instead of hardcoding values:

```python
# In config.py
class TrainingConfig:
    NUM_EPOCHS = 50          # Increase training duration
    BATCH_SIZE = 32          # Larger batches
    LEARNING_RATE = 5e-4     # Different learning rate

# In your script
from config import Config
print(f"Epochs: {Config.training.NUM_EPOCHS}")
print(f"Batch size: {Config.training.BATCH_SIZE}")
```

## Troubleshooting

### Issue: "No datasets found"

**Solution**: Ensure dataset paths are correct in `config.py`:

```python
class DatasetConfig:
    ROOT_DIR: str = "path/to/datasets"  # Update this
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in `config.py`:

```python
class TrainingConfig:
    BATCH_SIZE = 8  # Was 16
```

### Issue: "Very slow inference on CPU"

**Solution**: Use pre-optimized checkpoint or try CUDA/MPS:

```python
from utils import get_device
device = get_device(use_cuda=True)  # Use GPU if available
```

## Next Steps

- ✅ Read the [full README](README.md)
- ✅ Check out [examples](examples/) directory
- ✅ Run the [test suite](tests/)
- ✅ Try [fine-tuning](src/training/finetune_ocr.py)
- ✅ Build with [Gradio UI](src/inference/demo_gradio.py)

## API Reference

### Basic Inference

```python
from src.inference.test_ocr import OCRInference
from src.dataloader.ocr_dataloader import SmartNotesOCRDataset

# Load model
inference = OCRInference()

# Load data
dataset = SmartNotesOCRDataset(mode='val')
img, label = dataset[0]

# Predict
text = inference.predict(img)
```

### Training

```python
from src.training.train_ocr import OCRTrainer
from torch.utils.data import DataLoader

# Setup
trainer = OCRTrainer()
trainer.setup(num_classes=36)

# Train
trainer.train(train_loader, val_loader, num_epochs=20)
```

### Tokenization

```python
from src.dataloader.ocr_dataloader import TextTokenizer

tokenizer = TextTokenizer()

# Encode text
encoded = tokenizer.encode("hello")  # [7, 4, 11, 11, 14]

# Decode text
decoded = tokenizer.decode([7, 4, 11, 11, 14])  # "hello"
```

## Performance Tips

1. **Use GPU**: ~10x faster training
   ```python
   from utils import get_device
   device = get_device(use_cuda=True)
   ```

2. **Larger batches**: Better GPU utilization
   ```python
   BATCH_SIZE = 32  # 64 on high-end GPUs
   ```

3. **Data loading**: Increase workers
   ```python
   NUM_WORKERS = 4  # or 8 on systems with many cores
   ```

4. **Mixed precision**: For CUDA
   ```python
   # Enabled automatically in training script
   ```

## Getting Help

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/Kanike-tea/SmartNotes/issues)
- 💬 [Discussions](https://github.com/Kanike-tea/SmartNotes/discussions)
- 📧 [Contact Authors](mailto:contributors@smartnotes.local)

---

Ready to dive deeper? Start with the [full README](README.md)!
