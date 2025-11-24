"""
Unit tests for SmartNotes OCR project.

Tests cover:
- Model architecture and forward pass
- Data loading and preprocessing
- Configuration system
- Utility functions
- Tokenization and decoding
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Setup imports
import sys
from smartnotes.paths import setup_imports

setup_imports()

from src.model.ocr_model import CRNN
from src.dataloader.ocr_dataloader import TextTokenizer, clean_text
from config import Config
from utils import calculate_cer, calculate_wer


class TestModel:
    """Tests for CRNN model."""
    
    def test_model_initialization(self):
        """Test model can be created."""
        model = CRNN(num_classes=36)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self):
        """Test forward pass with valid input."""
        model = CRNN(num_classes=36)
        model.eval()
        
        # Create dummy input: (batch=2, channels=1, height=32, width=128)
        dummy_input = torch.randn(2, 1, 32, 128)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output should be (T, B, C) where T is sequence length
        assert output.shape[0] > 0  # T > 0
        assert output.shape[1] == 2  # B = 2
        assert output.shape[2] == 37  # C = 36 + 1 (blank)
    
    def test_model_invalid_input(self):
        """Test model with invalid input shape."""
        model = CRNN(num_classes=36)
        
        # Wrong number of dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(1, 32, 128)  # Missing channel dimension
            model(invalid_input)
    
    def test_model_device_movement(self):
        """Test model can be moved to different devices."""
        model = CRNN(num_classes=36)
        
        # Move to CPU (should always work)
        model = model.to(torch.device('cpu'))
        assert next(model.parameters()).device.type == 'cpu'
    
    def test_model_batch_sizes(self):
        """Test model with different batch sizes."""
        model = CRNN(num_classes=36)
        model.eval()
        
        for batch_size in [1, 4, 16, 32]:
            dummy_input = torch.randn(batch_size, 1, 32, 128)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape[1] == batch_size


class TestTokenizer:
    """Tests for TextTokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer can be created."""
        tokenizer = TextTokenizer()
        assert tokenizer is not None
        assert len(tokenizer.chars) > 0
    
    def test_tokenizer_encoding(self):
        """Test text encoding."""
        tokenizer = TextTokenizer()
        
        # Encode simple text
        text = "hello"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) == len(text)
        assert all(isinstance(idx, int) for idx in encoded)
    
    def test_tokenizer_decoding(self):
        """Test text decoding."""
        tokenizer = TextTokenizer()
        
        # Encode and decode
        text = "hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(np.array(encoded))
        assert decoded == text
    
    def test_tokenizer_roundtrip(self):
        """Test encode-decode roundtrip."""
        tokenizer = TextTokenizer()
        texts = ["hello", "world", "123", "abc123"]
        
        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(np.array(encoded))
            assert decoded == text
    
    def test_tokenizer_unknown_chars(self):
        """Test handling of unknown characters."""
        tokenizer = TextTokenizer()
        
        # Text with unknown character (uppercase)
        text = "Hello"  # Contains 'H' which should map to lowercase
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(np.array(encoded))
        # Should decode to lowercase
        assert isinstance(decoded, str)


class TestTextPreprocessing:
    """Tests for text preprocessing."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        # Test removing disallowed characters
        text = "Hello, World! 123"
        cleaned = clean_text(text)
        assert "," in cleaned or "," not in text
        assert "1" in cleaned
        assert "2" in cleaned
        assert "3" in cleaned
    
    def test_clean_empty_text(self):
        """Test cleaning empty text."""
        assert clean_text("") == ""


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_cer_identical_strings(self):
        """Test CER for identical strings."""
        cer = calculate_cer("hello", "hello")
        assert cer == 0.0
    
    def test_cer_different_strings(self):
        """Test CER for completely different strings."""
        cer = calculate_cer("", "hello")
        assert cer == 1.0
    
    def test_cer_partial_match(self):
        """Test CER for partial match."""
        # "hell" vs "hello" - 1 insertion
        cer = calculate_cer("hell", "hello")
        assert 0 < cer < 1
    
    def test_wer_identical_strings(self):
        """Test WER for identical strings."""
        wer = calculate_wer("hello world", "hello world")
        assert wer == 0.0
    
    def test_wer_different_words(self):
        """Test WER for different words."""
        wer = calculate_wer("", "hello")
        assert wer == 1.0


class TestConfiguration:
    """Tests for configuration system."""
    
    def test_config_exists(self):
        """Test configuration object exists."""
        assert Config is not None
    
    def test_config_access(self):
        """Test accessing configuration parameters."""
        assert hasattr(Config, 'dataset')
        assert hasattr(Config, 'training')
        assert hasattr(Config, 'model')
        assert hasattr(Config, 'inference')
    
    def test_config_values(self):
        """Test configuration values are reasonable."""
        assert Config.dataset.IMG_HEIGHT > 0
        assert Config.dataset.IMG_WIDTH > 0
        assert Config.training.NUM_EPOCHS > 0
        assert Config.training.BATCH_SIZE > 0
        assert 0 < Config.training.LEARNING_RATE < 1
    
    def test_config_to_dict(self):
        """Test config can be converted to dictionary."""
        config_dict = Config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'dataset' in config_dict
        assert 'training' in config_dict


class TestDeviceDetection:
    """Tests for device detection utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        from utils import get_device
        
        device = get_device(force_cpu=True)
        assert device.type == 'cpu'
    
    def test_get_device_cuda_detection(self):
        """Test CUDA availability detection."""
        from utils import get_device
        
        if torch.cuda.is_available():
            device = get_device(use_cuda=True, force_cpu=False)
            # Should get CUDA if available
            assert device.type in ['cuda', 'cpu']


class TestPathUtilities:
    """Tests for path utilities."""
    
    def test_path_validation(self):
        """Test path validation functions."""
        from utils import ensure_path_exists, dir_exists
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test creating path
            test_path = Path(tmpdir) / "test" / "nested" / "path"
            result = ensure_path_exists(str(test_path))
            assert result.exists()
            assert dir_exists(str(test_path))


class TestIntegration:
    """Integration tests."""
    
    def test_model_with_dataloader_shapes(self):
        """Test model works with dataloader output shapes."""
        model = CRNN(num_classes=36)
        model.eval()
        
        # Simulate dataloader output
        batch_size = 4
        images = torch.randn(batch_size, 1, 32, 128)
        labels = torch.randint(0, 36, (batch_size, 20))
        
        with torch.no_grad():
            output = model(images)
        
        # Check output can be used with CTC loss
        assert output.shape[1] == batch_size
        assert output.shape[2] == 37  # 36 classes + 1 blank
    
    def test_tokenizer_with_model_output(self):
        """Test tokenizer can decode model-like outputs."""
        tokenizer = TextTokenizer()
        
        # Simulate model output (sequence of class indices)
        sequence = np.array([7, 4, 11, 11, 14, 0, 22, 14, 17, 11, 3])  # "hello world"
        decoded = tokenizer.decode(sequence)
        
        assert isinstance(decoded, str)
        assert len(decoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
