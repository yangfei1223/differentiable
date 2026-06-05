"""Equirectangular 环境贴图 — 存储、采样、可导 mipmap 预滤波。"""
from __future__ import annotations

import math

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
        nn.Parameter [1, H, W, 3]
    """
    if init_image is not None:
        data = init_image.clone().float()
    else:
        # 均匀灰色 0.5, inverse-softplus: log(exp(x)-1)
        data = torch.ones(1, height, width, 3) * 0.5
        data = torch.log(torch.exp(data) - 1.0 + 1e-6)

    return nn.Parameter(data)


def _decode_env_map(raw: torch.Tensor) -> torch.Tensor:
    """Softplus 约束保证非负。"""
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


def sample_env_map(env_map: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """从环境贴图沿方向采样。

    Args:
        env_map: 原始环境贴图参数 [1, Eh, Ew, 3]
        dirs: 方向向量 [..., 3]

    Returns:
        颜色 [..., 3]
    """
    decoded = _decode_env_map(env_map)

    u, v = direction_to_equirect(dirs)
    orig_shape = u.shape
    n_pixels = u.numel()
    grid = torch.stack([u, v], dim=-1).reshape(1, 1, n_pixels, 2)

    tex = decoded.permute(0, 3, 1, 2)  # [1, 3, Eh, Ew]

    sampled = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.reshape(3, n_pixels).T

    return sampled.reshape(*orig_shape, 3)


# 缓存预计算的高斯核，避免每步重建
_gauss_kernel_cache: dict[tuple, torch.Tensor] = {}


def _get_gauss_kernel(n_levels: int, H: int, W: int, device: torch.device) -> list[torch.Tensor | None]:
    """预计算/缓存 n_levels 个高斯卷积核。

    Returns:
        长度 n_levels 的列表，level 0 为 None（不需要卷积），
        其余为 [3, 1, kH, kW] 的分组卷积核。
    """
    cache_key = (n_levels, H, W, device)
    if cache_key in _gauss_kernel_cache:
        return _gauss_kernel_cache[cache_key]

    kernels: list[torch.Tensor | None] = [None]  # level 0 = identity
    for level in range(1, n_levels):
        roughness = level / max(n_levels - 1, 1)
        sigma = roughness * min(H, W) * 0.25
        kernel_size = int(sigma * 4) | 1
        kernel_size = max(kernel_size, 3)
        kernel_size = min(kernel_size, min(H, W))
        if kernel_size % 2 == 0:
            kernel_size += 1

        k = kernel_size
        ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel_2d = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-8))
        kernel_2d = kernel_2d / kernel_2d.sum()
        # [3, 1, kH, kW] 用于 groups=3 的 conv2d
        kernel_3ch = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
        kernels.append(kernel_3ch)

    _gauss_kernel_cache[cache_key] = kernels
    return kernels


def prefilter_env_map(env_map: torch.Tensor, n_levels: int) -> torch.Tensor:
    """对环境贴图做可导的 2D 高斯卷积生成 mipmap 链。

    高斯核会被缓存，env_map 参与梯度计算。

    Args:
        env_map: 原始参数 [1, Eh, Ew, 3]
        n_levels: mipmap 级别数（包含 level 0）

    Returns:
        prefiltered [1, n_levels, Eh, Ew, 3]
    """
    decoded = _decode_env_map(env_map)
    H, W = decoded.shape[1], decoded.shape[2]

    kernels = _get_gauss_kernel(n_levels, H, W, decoded.device)

    levels = [decoded]
    inp = decoded[0].permute(2, 0, 1).unsqueeze(0)  # [1, 3, Eh, Ew]

    for level in range(1, n_levels):
        k = kernels[level]
        pad = k.shape[-1] // 2
        blurred = F.conv2d(inp, k, padding=pad, groups=3)  # [1, 3, Eh, Ew]
        blurred = blurred.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # [1, Eh, Ew, 3]
        levels.append(blurred)

    return torch.stack(levels, dim=1)


def sample_prefiltered(
    prefiltered: torch.Tensor,
    dirs: torch.Tensor,
    roughness: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """从预滤波 mipmap 按 roughness 采样。

    在两个相邻 mipmap 级别之间做线性插值。
    向量化实现：按 mip level 分组批量 grid_sample。

    Args:
        prefiltered: [1, n_levels, Eh, Ew, 3]
        dirs: 方向 [..., 3]
        roughness: [..., 1] 值域 [0, 1]
        n_levels: mipmap 级别数

    Returns:
        颜色 [..., 3]
    """
    u, v = direction_to_equirect(dirs)
    orig_shape = u.shape
    n_pixels = u.numel()
    device = prefiltered.device

    mip_level = roughness.reshape(-1) * (n_levels - 1)
    mip_level = mip_level.clamp(0, n_levels - 1)

    level_lo = mip_level.floor().long().clamp(0, n_levels - 1)
    level_hi = (level_lo + 1).clamp(max=n_levels - 1)
    frac = (mip_level - level_lo.float()).reshape(-1, 1)  # [N, 1]

    grid_flat = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)  # [N, 2]

    # 为每个 mip level 对收集像素索引，批量 grid_sample
    colors_lo = torch.zeros(n_pixels, 3, device=device)
    colors_hi = torch.zeros(n_pixels, 3, device=device)

    for lvl in range(n_levels):
        # lo == lvl 的像素
        mask_lo = (level_lo == lvl)
        if mask_lo.any():
            idx_lo = mask_lo.nonzero(as_tuple=True)[0]
            g = grid_flat[idx_lo].reshape(1, 1, -1, 2)
            tex = prefiltered[0, lvl].permute(2, 0, 1).unsqueeze(0)  # [1, 3, Eh, Ew]
            c = F.grid_sample(tex, g, mode="bilinear", padding_mode="border", align_corners=True)
            colors_lo[idx_lo] = c.reshape(3, -1).T

        # hi == lvl 的像素
        mask_hi = (level_hi == lvl)
        if mask_hi.any():
            idx_hi = mask_hi.nonzero(as_tuple=True)[0]
            g = grid_flat[idx_hi].reshape(1, 1, -1, 2)
            tex = prefiltered[0, lvl].permute(2, 0, 1).unsqueeze(0)
            c = F.grid_sample(tex, g, mode="bilinear", padding_mode="border", align_corners=True)
            colors_hi[idx_hi] = c.reshape(3, -1).T

    color = colors_lo * (1.0 - frac) + colors_hi * frac

    return color.reshape(*orig_shape, 3)
