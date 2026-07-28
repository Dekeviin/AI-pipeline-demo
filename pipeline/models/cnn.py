"""1-D CNN encoder over the feature window.

Input:  (batch, n_features, lookback) — each indicator is a channel, the CNN
convolves across time. Depth is config-driven: every entry in `channels` adds
a Conv→BatchNorm→ReLU block, so deepening the encoder is a YAML edit.
"""
import torch
import torch.nn as nn

from . import encoder


@encoder("cnn")
class CNNEncoder(nn.Module):
    def __init__(self, n_features: int, lookback: int, channels: list[int],
                 kernel: int = 3, embed_dim: int = 64):
        super().__init__()
        blocks, in_ch = [], n_features
        for out_ch in channels:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.conv(x)).squeeze(-1)   # (B, C)
        return torch.relu(self.proj(z))           # (B, embed_dim)
