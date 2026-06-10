"""PBR 材质参数化 — 单张 8 通道纹理 (base_color 3 + roughness 1 + metallic 1 + normal_xyz 3)。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# 通道布局: [base_color_R, base_color_G, base_color_B, roughness, metallic, normal_x, normal_y, normal_z]
N_CHANNELS = 8


def init_material_texture(resolution: int) -> nn.Parameter:
    """初始化材质贴图。

    8 通道:
    - [0:3] base_color — sigmoid 解码, 初始 0.5
    - [3:4] roughness — sigmoid 解码, 初始 0.5
    - [4:5] metallic — sigmoid 解码, 初始 ~0 (sigmoid(-5)≈0.007)
    - [5:8] normal_xyz — F.normalize 解码, 初始 (0, 0, 1)

    Args:
        resolution: 纹理分辨率 (正方形)。

    Returns:
        nn.Parameter [1, resolution, resolution, 8]
    """
    # ch 0-4: sigmoid inverse
    # roughness 初始 ~0.12 (sigmoid(-2)), metallic 初始 ~0.007 (sigmoid(-5))
    init_vals = torch.tensor([0.0, 0.0, 0.0, -2.0, -5.0, 0.0, 0.0, 1.0])
    data = init_vals.reshape(1, 1, 1, 8).expand(1, resolution, resolution, 8).clone()

    return nn.Parameter(data.float())


def decode_material(
    raw_texture: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从原始纹理张量解码材质参数。

    Args:
        raw_texture: 原始纹理 [1, H, W, 8]。

    Returns:
        (base_color, roughness, metallic, normal) —
        base_color [1, H, W, 3], roughness [1, H, W, 1],
        metallic [1, H, W, 1], normal [1, H, W, 3] (单位向量)
    """
    # ch 0-4: sigmoid 解码
    decoded = torch.sigmoid(raw_texture[..., :5])

    base_color = decoded[..., :3]   # [1, H, W, 3]
    roughness = decoded[..., 3:4]   # [1, H, W, 1]
    metallic = decoded[..., 4:5]    # [1, H, W, 1]

    # ch 5-7: normalize 解码 → 单位向量
    normal_raw = raw_texture[..., 5:8]  # [1, H, W, 3]
    normal = F.normalize(normal_raw, dim=-1)

    return base_color, roughness, metallic, normal


def compute_F0(
    base_color: torch.Tensor,
    metallic: torch.Tensor,
    dielectric_F0: float = 0.04,
) -> torch.Tensor:
    """计算菲涅尔 F0。

    F0 = lerp(dielectric_F0, base_color, metallic)
    """
    return dielectric_F0 * (1.0 - metallic) + base_color * metallic
