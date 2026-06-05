"""测试 PBR 材质参数化。"""
import torch
from src.shading.pbr.material import (
    init_material_texture,
    decode_material,
    compute_F0,
)


def test_init_material_texture_shape():
    tex = init_material_texture(64)
    assert tex.shape == (1, 64, 64, 5)


def test_init_material_texture_is_parameter():
    tex = init_material_texture(64)
    assert isinstance(tex, torch.nn.Parameter)


def test_decode_material_base_color_range():
    tex = init_material_texture(32)
    base_color, roughness, metallic = decode_material(tex)
    assert base_color.min() >= 0.0
    assert base_color.max() <= 1.0
    assert roughness.min() >= 0.0
    assert roughness.max() <= 1.0
    assert metallic.min() >= 0.0
    assert metallic.max() <= 1.0


def test_decode_material_shapes():
    tex = init_material_texture(16)
    base_color, roughness, metallic = decode_material(tex)
    assert base_color.shape == (1, 16, 16, 3)
    assert roughness.shape == (1, 16, 16, 1)
    assert metallic.shape == (1, 16, 16, 1)


def test_compute_F0_dielectric():
    base_color = torch.ones(1, 4, 4, 3) * 0.5
    metallic = torch.zeros(1, 4, 4, 1)
    F0 = compute_F0(base_color, metallic)
    assert torch.allclose(F0, torch.ones(1, 4, 4, 3) * 0.04, atol=1e-5)


def test_compute_F0_metallic():
    base_color = torch.ones(1, 4, 4, 3) * 0.8
    metallic = torch.ones(1, 4, 4, 1)
    F0 = compute_F0(base_color, metallic)
    assert torch.allclose(F0, base_color, atol=1e-5)


def test_material_gradient_flows():
    tex = init_material_texture(16)
    base_color, roughness, metallic = decode_material(tex)
    loss = base_color.sum() + roughness.sum() + metallic.sum()
    loss.backward()
    assert tex.grad is not None
    assert tex.grad.abs().sum() > 0
