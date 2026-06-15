"""Feature map initialization for Neural Lightmap.

Each submesh owns a learnable feature texture [1, H, W, C] that implicitly
encodes albedo, normals, AO, and incoming radiance.
"""
from __future__ import annotations

import torch


def init_feature_map(
    resolution: int,
    feature_dim: int = 12,
    init_std: float = 0.1,
) -> torch.Tensor:
    """Create a randomly initialized feature texture.

    Args:
        resolution: texture H/W.
        feature_dim: feature channels C.
        init_std: initialization standard deviation (small to start near flat).

    Returns:
        Tensor [1, resolution, resolution, feature_dim], float32.
    """
    return torch.randn(1, resolution, resolution, feature_dim) * init_std
