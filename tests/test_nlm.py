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


def test_init_feature_map_shape():
    """init_feature_map returns [1, res, res, C] tensor."""
    from src.shading.nlm.feature_map import init_feature_map

    fm = init_feature_map(resolution=64, feature_dim=12, init_std=0.1)
    assert fm.shape == (1, 64, 64, 12)
    assert fm.dtype == torch.float32


def test_init_feature_map_std():
    """Init std approximately matches configured value."""
    from src.shading.nlm.feature_map import init_feature_map

    fm = init_feature_map(resolution=512, feature_dim=12, init_std=0.1)
    # randn * 0.1 → std ≈ 0.1
    assert 0.08 < fm.std().item() < 0.12


def test_nlm_config_defaults():
    """NeuralLightmapConfig has correct defaults."""
    from src.config import NeuralLightmapConfig

    cfg = NeuralLightmapConfig()
    assert cfg.feature_dim == 12
    assert cfg.pe_level == 2
    assert cfg.mlp_hidden_dim == 32
    assert cfg.feature_lr == 0.1
    assert cfg.mlp_lr == 0.001
    assert cfg.feature_tv_weight == 0.00001
    assert cfg.feature_init_std == 0.1


def test_config_load_nlm_yaml(tmp_path):
    """YAML with 'nlm' section parses correctly."""
    from src.config import load_config

    yaml_content = """
render_mode: nlm
nlm:
  feature_dim: 16
  pe_level: 3
  feature_lr: 0.05
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(str(p))
    assert cfg.render_mode == "nlm"
    assert cfg.nlm.feature_dim == 16
    assert cfg.nlm.pe_level == 3
    assert cfg.nlm.feature_lr == 0.05
