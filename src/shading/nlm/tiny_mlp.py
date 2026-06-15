"""TinyMLP — lightweight decoder for Neural Lightmap.

Maps (feature ⊕ view_pe) → RGB radiance. Output uses Softplus to allow
HDR values > 1.0 while remaining non-negative.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """3-layer MLP with Softplus output for HDR radiance."""

    def __init__(self, in_dim: int = 27, hidden_dim: int = 32, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
