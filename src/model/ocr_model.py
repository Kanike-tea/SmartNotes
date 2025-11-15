import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Handwriting Recognition (BLSTM + CTC)
    Compatible with pre-trained checkpoints (no height reduction).
    """

    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # --- CNN backbone (height -> 2) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        # Height after CNN = 2, so RNN input = 512 * 2 = 1024
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Output layer (+1 for CTC blank)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, width, channels, height)
        x = x.view(b, w, c * h)  # (batch, seq_len=width, feature=c*h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)  # (T, B, C)
