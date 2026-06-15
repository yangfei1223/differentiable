"""Positional encoding for view directions (NeRF-style).

γ(d) = [d, sin(2^0 π d), cos(2^0 π d), ..., sin(2^{L-1} π d), cos(2^{L-1} π d)]

Output dim = 3 + 2*L*3 = 3*(1 + 2*L).
L=2 → 15D, L=3 → 21D.
"""
from __future__ import annotations

import torch


def positional_encode(d: torch.Tensor, level: int = 2) -> torch.Tensor:
    """Apply NeRF-style positional encoding to a direction vector.

    Args:
        d: direction tensor, last dim must be 3. Shape [..., 3].
        level: PE frequency levels L.

    Returns:
        Encoded tensor of shape [..., 3*(1 + 2*level)].
    """
    freqs = 2.0 ** torch.arange(level, device=d.device, dtype=d.dtype)  # [L]
    # Outer product: freqs × d → [L, ..., 3] via broadcast
    # We want sin(2^k * pi * d) for each k
    scaled = d.unsqueeze(-2) * (freqs * torch.pi).view(*([1] * (d.dim() - 1)), -1, 1)
    # scaled: [..., L, 3]
    sin = torch.sin(scaled)
    cos = torch.cos(scaled)
    # Interleave sin/cos per frequency level: [..., L, 2, 3]
    sin_cos = torch.stack([sin, cos], dim=-2)
    # Flatten last three dims (L, 2, 3) → [..., L*6]
    sin_cos_flat = sin_cos.flatten(start_dim=-3)
    return torch.cat([d, sin_cos_flat], dim=-1)
