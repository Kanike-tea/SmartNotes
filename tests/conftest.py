"""
Test fixtures and configuration for SmartNotes tests.
"""

import pytest
import torch
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def device():
    """Provide a device for testing."""
    return torch.device('cpu')


@pytest.fixture
def dummy_image():
    """Provide a dummy image tensor."""
    return torch.randn(1, 1, 32, 128)


@pytest.fixture
def dummy_batch():
    """Provide a dummy batch of images."""
    batch_size = 4
    images = torch.randn(batch_size, 1, 32, 128)
    labels = torch.randint(0, 36, (batch_size, 20))
    return images, labels


@pytest.fixture
def sample_texts():
    """Provide sample text strings."""
    return [
        "hello",
        "world",
        "123",
        "abc123def456",
        "a",
    ]
