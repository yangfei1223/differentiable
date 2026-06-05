"""固定 BRDF LUT — GGX BRDF 积分表生成与采样。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _ggx_importance_sample(NdotV: float, roughness: float, num_samples: int = 512) -> tuple[float, float]:
    """对 GGX BRDF 做 importance sampling 积分，返回 (scale, bias)。"""
    V = NdotV
    a = roughness * roughness
    a2 = a * a

    scale = 0.0
    bias = 0.0

    for i in range(num_samples):
        # Van der Corput radical inverse (Hammersley)
        bits = i
        bits = ((bits << 16) & 0xFFFF0000) | ((bits >> 16) & 0x0000FFFF)
        bits = ((bits << 8) & 0xFF00FF00) | ((bits >> 8) & 0x00FF00FF)
        bits = ((bits << 4) & 0xF0F0F0F0) | ((bits >> 4) & 0x0F0F0F0F)
        bits = ((bits << 2) & 0xCCCCCCCC) | ((bits >> 2) & 0x33333333)
        bits = ((bits << 1) & 0xAAAAAAAA) | ((bits >> 1) & 0x55555555)
        xi_1 = float(bits) / float(0x100000000)

        bits2 = i + 1
        bits2 = ((bits2 << 16) & 0xFFFF0000) | ((bits2 >> 16) & 0x0000FFFF)
        bits2 = ((bits2 << 8) & 0xFF00FF00) | ((bits2 >> 8) & 0x00FF00FF)
        bits2 = ((bits2 << 4) & 0xF0F0F0F0) | ((bits2 >> 4) & 0x0F0F0F0F)
        bits2 = ((bits2 << 2) & 0xCCCCCCCC) | ((bits2 >> 2) & 0x33333333)
        bits2 = ((bits2 << 1) & 0xAAAAAAAA) | ((bits2 >> 1) & 0x55555555)
        xi_2 = float(bits2) / float(0x100000000)

        a2_safe = max(a2, 1e-7)
        phi = 2.0 * math.pi * xi_1
        cos_theta = math.sqrt((1.0 - xi_2) / (1.0 + (a2_safe - 1.0) * xi_2))
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        Hx = sin_theta * math.cos(phi)
        Hy = sin_theta * math.sin(phi)
        Hz = cos_theta

        Vx = math.sqrt(max(1.0 - V * V, 0.0))
        Vz = V

        NdotH = Hz
        VdotH = Vz * Hz + Vx * Hx
        VdotH = max(VdotH, 1e-7)

        k = (roughness + 1) ** 2 / 8.0
        G_V = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
        G_L = cos_theta / (cos_theta * (1.0 - k) + k + 1e-7)
        G = G_V * G_L

        F_weight = G * VdotH / (NdotH * NdotV + 1e-7)

        scale += F_weight
        bias += F_weight * (1.0 - (1.0 - VdotH) ** 5)

    scale /= num_samples
    bias /= num_samples

    return scale, bias


def generate_brdf_lut(size: int = 256) -> torch.Tensor:
    """生成 GGX BRDF 积分查找表。

    Args:
        size: LUT 分辨率 (正方形)。

    Returns:
        Tensor [size, size, 2]: 通道 0 = scale, 通道 1 = bias。
    """
    lut = torch.zeros(size, size, 2)

    for y in range(size):
        roughness = (y + 0.5) / size
        roughness = max(roughness, 0.04)

        for x in range(size):
            NdotV = (x + 0.5) / size
            NdotV = max(NdotV, 0.001)

            scale, bias = _ggx_importance_sample(NdotV, roughness, num_samples=512)
            lut[y, x, 0] = scale
            lut[y, x, 1] = bias

    return lut


def sample_brdf(
    lut: torch.Tensor,
    NdotV: torch.Tensor,
    roughness: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 BRDF LUT 采样。

    Args:
        lut: [size, size, 2]
        NdotV: [...] 值域 [0, 1]
        roughness: [...] 值域 [0, 1]

    Returns:
        (scale, bias) — 各为 [...] 形状
    """
    size = lut.shape[0]
    device = lut.device

    u = NdotV.clamp(0, 1)
    v = roughness.clamp(0, 1)

    orig_shape = u.shape
    n = u.numel()
    grid = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1).reshape(1, 1, n, 2)

    lut_tex = lut.permute(2, 0, 1).unsqueeze(0).to(device)

    sampled = F.grid_sample(
        lut_tex, grid.to(device),
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    sampled = sampled.reshape(2, n).T

    scale = sampled[:, 0].reshape(*orig_shape)
    bias = sampled[:, 1].reshape(*orig_shape)

    return scale, bias
