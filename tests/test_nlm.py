"""Neural Lightmap unit tests."""
import torch
import pytest


def test_positional_encode_shape_and_range():
    """L=2 produces 15D output (3 raw + 4 freq bands * 3 dims)."""
    from src.shading.nlm.positional_encode import positional_encode

    d = torch.tensor([[[[0.3, -0.5, 0.8]]]])  # [1,1,1,3]
    out = positional_encode(d, level=2)
    assert out.shape == (1, 1, 1, 15)


def test_positional_encode_batch():
    """Works on flat batched input [N,3]."""
    from src.shading.nlm.positional_encode import positional_encode

    d = torch.randn(100, 3)
    out = positional_encode(d, level=2)
    assert out.shape == (100, 15)


def test_positional_encode_zero_input():
    """Zero input produces [0,0,0,sin=0...,cos=1...] — sin(0)=0, cos(0)=1."""
    from src.shading.nlm.positional_encode import positional_encode

    d = torch.zeros(1, 3)
    out = positional_encode(d, level=2)
    # First 3 channels: raw d = 0
    assert torch.allclose(out[0, :3], torch.zeros(3))
    # Next 3 channels: sin(2^0 * pi * 0) = sin(0) = 0
    assert torch.allclose(out[0, 3:6], torch.zeros(3), atol=1e-6)
    # Next 3 channels: cos(2^0 * pi * 0) = cos(0) = 1
    assert torch.allclose(out[0, 6:9], torch.ones(3), atol=1e-6)


def test_tiny_mlp_shape():
    """TinyMLP maps 27D → 3D."""
    from src.shading.nlm.tiny_mlp import TinyMLP

    mlp = TinyMLP(in_dim=27, hidden_dim=32, out_dim=3)
    x = torch.randn(10, 27)
    out = mlp(x)
    assert out.shape == (10, 3)


def test_tiny_mlp_non_negative_output():
    """Softplus output is non-negative (HDR radiance ≥ 0)."""
    from src.shading.nlm.tiny_mlp import TinyMLP

    mlp = TinyMLP(in_dim=27, hidden_dim=32, out_dim=3)
    x = torch.randn(100, 27) * 10  # extreme inputs
    out = mlp(x)
    assert (out >= 0).all(), "Softplus output must be non-negative"


def test_tiny_mlp_param_count():
    """~2K params (27*32 + 32 + 32*32 + 32 + 32*3 + 3 = 2019)."""
    from src.shading.nlm.tiny_mlp import TinyMLP

    mlp = TinyMLP(in_dim=27, hidden_dim=32, out_dim=3)
    n = sum(p.numel() for p in mlp.parameters())
    assert 1500 < n < 3500, f"Expected ~2K params, got {n}"
