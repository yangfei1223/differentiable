"""Equirectangular 环境贴图 — 高斯预滤波 mipmap + nvdiffrast dr.texture 采样。"""
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


# ---------------------------------------------------------------------------
# 高斯卷积核缓存
# ---------------------------------------------------------------------------
_kernel_cache: dict[tuple, torch.Tensor] = {}


def _make_gauss_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """生成 2D 高斯卷积核 [1, 1, k, k]（广播用）。"""
    k = int(sigma * 4) | 1
    k = max(k, 3)
    cache_key = (k, sigma, device)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-8))
    kernel = kernel / kernel.sum()
    _kernel_cache[cache_key] = kernel
    return kernel


def prefilter_mipmap(env_map_raw: torch.Tensor, max_mip: int) -> list[torch.Tensor]:
    """对解码后的 env map 生成高斯预滤波 mipmap 链。

    不用 nvdiffrast 内置 box filter，而是每级用高斯卷积。
    sigma 随 mip level 递增，模拟 roughness 增大时的环境模糊。

    Args:
        env_map_raw: 原始参数 [1, Eh, Ew, 3]
        max_mip: mipmap 级别数

    Returns:
        list of [1, Eh, Ew, 3] — 从 mip 1 到 mip (max_mip-1)
        （mip 0 就是原始纹理，dr.texture 的 tex 参数）
    """
    decoded = _decode_env_map(env_map_raw)  # [1, Eh, Ew, 3]
    H, W = decoded.shape[1], decoded.shape[2]
    device = decoded.device

    mip_list: list[torch.Tensor] = []
    inp = decoded[0].permute(2, 0, 1).unsqueeze(0)  # [1, 3, Eh, Ew]

    for level in range(1, max_mip):
        roughness = level / max(max_mip - 1, 1)
        # sigma 与 env map 尺寸和 roughness 成正比
        sigma = roughness * min(H, W) * 0.25
        sigma = max(sigma, 0.5)

        kernel = _make_gauss_kernel(sigma, device)
        pad = kernel.shape[0] // 2
        # 分组卷积: [1, 3, H, W] × [3, 1, kH, kW] → [1, 3, H, W]
        k3 = kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
        blurred = F.conv2d(inp, k3, padding=pad, groups=3)
        mip_list.append(blurred.squeeze(0).permute(1, 2, 0).unsqueeze(0))  # [1, Eh, Ew, 3]

    return mip_list


def sample_env_map(
    env_map: torch.Tensor,
    dirs: torch.Tensor,
    mip_level_bias: torch.Tensor | None = None,
    custom_mip: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """从环境贴图沿方向采样，使用 nvdiffrast dr.texture。

    Args:
        env_map: 原始环境贴图参数 [1, Eh, Ew, 3]
        dirs: 方向向量 [..., 3]
        mip_level_bias: 可选 mip level 偏置 [...]。
            None = mip 0 (最清晰)，值越大越模糊。
        custom_mip: 可选自定义 mipmap 链（prefilter_mipmap 的输出）。
            若提供则替代 nvdiffrast 内置 box mipmap。

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
    grid = grid.reshape(1, 1, -1, 2)    # [1, 1, N, 2]

    kwargs = dict(
        filter_mode="linear",
        boundary_mode="wrap",
    )

    if mip_level_bias is not None:
        kwargs["filter_mode"] = "linear-mipmap-linear"
        kwargs["max_mip_level"] = max_mip
        bias = mip_level_bias.reshape(1, 1, -1)
        kwargs["mip_level_bias"] = bias
        if custom_mip is not None:
            kwargs["mip"] = custom_mip

    color = dr.texture(decoded, grid, **kwargs)  # [1, 1, N, 3]
    return color.reshape(*orig_shape, 3)
