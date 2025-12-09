import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        logits = self.fc(h)
        return logits


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        pad = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2, c3, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length) or (batch, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.features(x).squeeze(-1)
        feats = self.dropout(feats)
        logits = self.classifier(feats)
        return logits
