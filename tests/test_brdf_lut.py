"""测试 BRDF LUT。"""
import torch
from src.shading.pbr.brdf_lut import generate_brdf_lut, sample_brdf


def test_generate_brdf_lut_shape():
    lut = generate_brdf_lut(64)
    assert lut.shape == (64, 64, 2)


def test_generate_brdf_lut_range():
    lut = generate_brdf_lut(64)
    assert lut.min() >= 0.0
    assert lut.max() <= 1.5


def test_sample_brdf_shape():
    lut = generate_brdf_lut(64)
    NdotV = torch.tensor([0.5, 0.8, 0.1])
    roughness = torch.tensor([0.3, 0.7, 0.0])
    scale, bias = sample_brdf(lut, NdotV, roughness)
    assert scale.shape == (3,)
    assert bias.shape == (3,)


def test_sample_brdf_perfect_mirror():
    """roughness=0, NdotV=1 的 scale 应接近 1。"""
    lut = generate_brdf_lut(256)
    NdotV = torch.tensor([1.0])
    roughness = torch.tensor([0.0])
    scale, bias = sample_brdf(lut, NdotV, roughness)
    assert scale.item() > 0.8
