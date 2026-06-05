"""PBR 材质参数化 — 单张 5 通道纹理 + sigmoid 约束。"""
from __future__ import annotations

import torch
import torch.nn as nn


def init_material_texture(resolution: int) -> nn.Parameter:
    """初始化材质贴图。

    5 通道: [base_color_R, base_color_G, base_color_B, roughness, metallic]
    存储为 nn.Parameter，初始值经过 inverse-sigmoid 映射以使得
    sigmoid(raw) ≈ 期望初始值。

    Args:
        resolution: 纹理分辨率 (正方形)。

    Returns:
        nn.Parameter [1, resolution, resolution, 5]
    """
    # sigmoid_inv(0.5) = 0.0, sigmoid_inv(0.0) → -∞, 用 -5.0 近似 (sigmoid(-5)≈0.007)
    init_vals = torch.tensor([0.0, 0.0, 0.0, 0.0, -5.0])  # [5]
    data = init_vals.reshape(1, 1, 1, 5).expand(1, resolution, resolution, 5).clone()

    return nn.Parameter(data.float())


def decode_material(
    raw_texture: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从原始纹理张量解码材质参数。

    Args:
        raw_texture: 原始纹理 [1, H, W, 5]。

    Returns:
        (base_color, roughness, metallic) —
        base_color [1, H, W, 3], roughness [1, H, W, 1], metallic [1, H, W, 1]
    """
    decoded = torch.sigmoid(raw_texture)  # [1, H, W, 5]

    base_color = decoded[..., :3]  # [1, H, W, 3]
    roughness = decoded[..., 3:4]  # [1, H, W, 1]
    metallic = decoded[..., 4:5]  # [1, H, W, 1]

    return base_color, roughness, metallic


def compute_F0(
    base_color: torch.Tensor,
    metallic: torch.Tensor,
    dielectric_F0: float = 0.04,
) -> torch.Tensor:
    """计算菲涅尔 F0。

    F0 = lerp(dielectric_F0, base_color, metallic)

    Args:
        base_color: [1, H, W, 3]
        metallic: [1, H, W, 1]
        dielectric_F0: 非金属 F0 默认值。

    Returns:
        F0 [1, H, W, 3]
    """
    return dielectric_F0 * (1.0 - metallic) + base_color * metallic
