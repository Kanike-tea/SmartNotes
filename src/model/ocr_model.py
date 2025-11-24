"""
CRNN model for Handwritten Text Recognition.

This module implements a Convolutional Recurrent Neural Network (CRNN) that
combines CNN for feature extraction and LSTM for sequence modeling, optimized
for handwriting recognition with CTC loss.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Handwriting Recognition.
    
    Architecture:
    - CNN backbone: Extracts spatial features from images
    - LSTM layers: Bidirectional LSTM for sequence modeling
    - FC layer: Maps to character predictions for CTC loss
    
    The model is designed to work with 32x128 pixel images and output
    character-level predictions compatible with CTC loss.
    
    Attributes:
        num_classes: Number of character classes in the vocabulary
        cnn: Sequential CNN backbone
        rnn: Bidirectional LSTM layer
        fc: Fully connected output layer
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialize the CRNN model.
        
        Args:
            num_classes: Number of character classes (vocabulary size).
                        The actual output will have num_classes + 1
                        (including CTC blank token).
        
        Raises:
            ValueError: If num_classes is less than 1
        """
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        
        super(CRNN, self).__init__()
        self.num_classes = num_classes

        # --- CNN backbone (height: 32 -> 2) ---
        # Each conv layer with stride 1, kernel 3x3, padding 1
        # MaxPool reduces spatial dimensions
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 64 channels, height 32 -> 16
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            # Block 2: 64 -> 128 channels, height 16 -> 8
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            # Block 3: 128 -> 256 channels, height 8 -> 8
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: 256 -> 256 channels, height 8 -> 8
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # Height: 8 -> 4

            # Block 5: 256 -> 512 channels, height 4 -> 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))   # Height: 4 -> 2
        )

        # Height after CNN = 2, so RNN input = 512 * 2 = 1024
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Output layer: 512 (256*2 for bidirectional) -> num_classes + 1 (for CTC blank)
        self.fc = nn.Linear(512, num_classes + 1)
        
        # Validation assertion (CRITICAL for debugging)
        # Ensure output shape matches expected CTC format
        expected_output_classes = num_classes + 1
        actual_output_classes = self.fc.out_features
        assert actual_output_classes == expected_output_classes, \
            f"Model output mismatch: expected {expected_output_classes} classes, " \
            f"got {actual_output_classes}. This will cause CTC loss mismatch!"
        
        # Log model info
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[CRNN] Initialized: {num_classes} chars + 1 blank = "
                   f"{actual_output_classes} output classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width).
               Typically (batch, 1, 32, 128) for standard 32x128 images.
        
        Returns:
            Tensor of shape (T, B, num_classes+1) where:
            - T is the sequence length (width dimension after feature extraction)
            - B is the batch size
            - num_classes+1 is the output vocabulary size including CTC blank
            
            This format is compatible with nn.CTCLoss which expects
            (T, B, C) shaped output.
        
        Raises:
            ValueError: If input shape is invalid
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        # CNN feature extraction: (B, 1, 32, 128) -> (B, 512, 2, W')
        x = self.cnn(x)
        
        # Get dimensions
        b, c, h, w = x.size()  # B: batch, C: channels (512), H: height (2), W: width
        
        # Permute for sequence input: (B, 512, 2, W') -> (B, W', 512, 2)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Reshape to (B, W', 512*2): sequence of vectors
        # W' is the sequence length, 512*2=1024 is the feature dimension
        x = x.view(b, w, c * h)
        
        # LSTM: (B, T, 1024) -> (B, T, 512)
        x, _ = self.rnn(x)
        
        # FC layer: (B, T, 512) -> (B, T, num_classes+1)
        x = self.fc(x)
        
        # Transpose for CTC loss: (B, T, C) -> (T, B, C)
        # CTC loss expects (T, B, C) where T is sequence length
        return x.permute(1, 0, 2)


def create_model(num_classes: int, device: torch.device = None) -> CRNN:
    """
    Factory function to create and initialize a CRNN model.
    
    Args:
        num_classes: Number of character classes
        device: Device to place model on (cpu, cuda, mps, etc.)
        
    Returns:
        CRNN model instance on the specified device
    """
    model = CRNN(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model initialization and forward pass
    model = CRNN(num_classes=36)
    model.eval()
    
    # Test with dummy input
    dummy_input = torch.randn(2, 1, 32, 128)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (T={dummy_input.size(3)//4}, B=2, C=37)")

