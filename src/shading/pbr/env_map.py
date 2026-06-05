"""Equirectangular 环境贴图 — 使用 nvdiffrast dr.texture 采样与 mipmap。"""
from __future__ import annotations

import math

import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_env_map(height: int, width: int, init_image: torch.Tensor | None = None) -> nn.Parameter:
    """初始化环境贴图。

    Args:
        height: 贴图高度。
        width: 贴图宽度。
        init_image: 可选初始图像 [1, H, W, 3]。None 则用均匀灰色。

    Returns:
        nn.Parameter [1, H, W, 3]（存储为 raw 值，经 softplus 解码后为 HDR ≥ 0）
    """
    if init_image is not None:
        data = init_image.clone().float()
    else:
        # 均匀灰色 0.5 → softplus_inv(0.5) = log(exp(0.5)-1) ≈ -0.193
        data = torch.ones(1, height, width, 3) * (math.log(math.exp(0.5) - 1.0 + 1e-6))

    return nn.Parameter(data)


def _decode_env_map(raw: torch.Tensor) -> torch.Tensor:
    """softplus 约束保证 HDR 非负。配合 L2 正则化防止值爆炸。"""
    return F.softplus(raw)


def direction_to_equirect(dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将方向向量转换为 equirect UV 坐标。

    Args:
        dirs: 归一化方向 [..., 3] (x, y, z)

    Returns:
        (u, v) — 各为 [...] 形状，值域 [0, 1]
    """
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    u = torch.atan2(z, x) / (2.0 * math.pi) + 0.5
    v = torch.asin(y.clamp(-0.999, 0.999)) / math.pi + 0.5

    return u, v


def _compute_max_mip_level(H: int, W: int) -> int:
    """计算 env map 的最大 mip level 数。"""
    return int(math.floor(math.log2(max(H, W))))


def sample_env_map(env_map: torch.Tensor, dirs: torch.Tensor, mip_level_bias: torch.Tensor | None = None) -> torch.Tensor:
    """从环境贴图沿方向采样，使用 nvdiffrast dr.texture。

    Args:
        env_map: 原始环境贴图参数 [1, Eh, Ew, 3]
        dirs: 方向向量 [..., 3]
        mip_level_bias: 可选 mip level 偏置 [...]。
            None = mip 0 (最清晰)，值越大越模糊。
            传入 float 时直接作为绝对 mip level。

    Returns:
        颜色 [..., 3]
    """
    decoded = _decode_env_map(env_map)  # [1, Eh, Ew, 3]
    H, W = decoded.shape[1], decoded.shape[2]
    max_mip = _compute_max_mip_level(H, W)

    u, v = direction_to_equirect(dirs)
    orig_shape = u.shape

    # dr.texture 需要 [B, H, W, 2] 的 UV
    grid = torch.stack([u, v], dim=-1)  # [..., 2]
    # 展平到 [1, 1, N, 2]
    grid = grid.reshape(1, 1, -1, 2)

    kwargs = dict(
        filter_mode="linear",
        boundary_mode="clamp",
    )

    if mip_level_bias is not None:
        kwargs["filter_mode"] = "linear-mipmap-linear"
        kwargs["max_mip_level"] = max_mip
        # mip_level_bias 需要 [B, H, W] 形状
        bias = mip_level_bias.reshape(1, 1, -1)
        kwargs["mip_level_bias"] = bias

    color = dr.texture(decoded, grid, **kwargs)  # [1, 1, N, 3]
    return color.reshape(*orig_shape, 3)
