"""固定 BRDF LUT — GGX BRDF 积分表生成与采样。全 PyTorch 向量化。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _hammersley_seq(num_samples: int, device: str = "cpu") -> torch.Tensor:
    """生成 Hammersley 序列 [num_samples, 2]。"""
    i = torch.arange(num_samples, dtype=torch.float32, device=device)
    # Van der Corput radical inverse base 2
    bits = i.clone().long()
    bits = ((bits << 16) & 0xFFFF0000) | ((bits >> 16) & 0x0000FFFF)
    bits = ((bits << 8) & 0xFF00FF00) | ((bits >> 8) & 0x00FF00FF)
    bits = ((bits << 4) & 0xF0F0F0F0) | ((bits >> 4) & 0x0F0F0F0F)
    bits = ((bits << 2) & 0xCCCCCCCC) | ((bits >> 2) & 0x33333333)
    bits = ((bits << 1) & 0xAAAAAAAA) | ((bits >> 1) & 0x55555555)
    xi_2 = bits.float() / 0x100000000
    xi_1 = i / num_samples
    return torch.stack([xi_1, xi_2], dim=-1)  # [S, 2]


def generate_brdf_lut(size: int = 256, num_samples: int = 512) -> torch.Tensor:
    """生成 GGX BRDF 积分查找表（全 PyTorch 向量化）。

    Args:
        size: LUT 分辨率 (正方形)。
        num_samples: 每像素 importance sampling 样本数。

    Returns:
        Tensor [size, size, 2]: 通道 0 = scale, 通道 1 = bias。
    """
    device = "cpu"

    # Hammersley 序列 [S, 2]
    hammersley = _hammersley_seq(num_samples, device)
    xi_1 = hammersley[:, 0]  # [S]
    xi_2 = hammersley[:, 1]  # [S]

    # roughness / NdotV 网格: [size]
    roughness = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) / size
    roughness = roughness.clamp(min=0.04)
    ndotv = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) / size
    ndotv = ndotv.clamp(min=0.001)

    # 广播: roughness [size,1,1], ndotv [1,size,1], xi [1,1,S]
    R = roughness[:, None, None]       # [size, 1, 1]
    V = ndotv[None, :, None]           # [1, size, 1]
    X1 = xi_1[None, None, :]           # [1, 1, S]
    X2 = xi_2[None, None, :]           # [1, 1, S]

    a = R * R
    a2 = a * a
    a2_safe = a2.clamp(min=1e-7)
    k = (R + 1.0).pow(2) / 8.0

    # GGX importance sampling
    phi = 2.0 * math.pi * X1
    cos_theta = ((1.0 - X2) / (1.0 + (a2_safe - 1.0) * X2)).sqrt()
    sin_theta = (1.0 - cos_theta * cos_theta).sqrt()

    Hx = sin_theta * phi.cos()
    Hz = cos_theta

    # View direction (假设 Vz=NdotV, Vy=0, Vx=sqrt(1-V^2))
    Vx = (1.0 - V * V).clamp(min=0.0).sqrt()
    Vz = V

    # Geometric terms
    NdotH = Hz
    VdotH = (Vz * Hz + Vx * Hx).clamp(min=1e-7)

    G_V = V / (V * (1.0 - k) + k + 1e-7)
    G_L = cos_theta / (cos_theta * (1.0 - k) + k + 1e-7)
    G = G_V * G_L

    F_weight = G * VdotH / (NdotH * V + 1e-7)

    scale = F_weight.mean(dim=-1)  # [size, size]
    bias = (F_weight * (1.0 - (1.0 - VdotH).pow(5))).mean(dim=-1)  # [size, size]

    lut = torch.stack([scale, bias], dim=-1)  # [size, size, 2]
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
