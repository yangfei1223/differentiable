"""球谐（Spherical Harmonics）基函数评估、颜色解码与纹理初始化。

支持 order 0 / 1 / 2（共 1 + 3 + 5 = 9 个基函数，27 个 SH 系数用于 RGB 三通道）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# SH 常量
# ---------------------------------------------------------------------------

_C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
_C1 = 0.4886025119029199   # sqrt(3/(4*pi))
_C2 = [
    1.0925484305920792,   # sqrt(15/(4*pi)) / 2  (xy)
    0.31539156525252005,  # sqrt(15/(4*pi)) / 2 * ... (yz) 实际为 0.5*sqrt(15/pi)/2*...
    0.5462742152960396,   # (2zz - xx - yy) 系数
    0.31539156525252005,  # xz
    0.5900435899266435,   # xx - yy
]
# 精确值参考:
# _C2[0] = 0.5 * sqrt(15/pi) / 2  ≈ 1.0925484305920792
# _C2[1] = 0.5 * sqrt(15/pi) / 2  ≈ 0.31539156525252005  (实际是 sqrt(15/pi)/4 系数)
# _C2[2] = 0.25 * sqrt(5/pi)       ≈ 0.5462742152960396
# _C2[3] = 0.5 * sqrt(15/pi) / 2  ≈ 0.31539156525252005
# _C2[4] = 0.25 * sqrt(15/pi)      ≈ 0.5900435899266435


# ---------------------------------------------------------------------------
# eval_sh_basis
# ---------------------------------------------------------------------------

def eval_sh_basis(dirs: torch.Tensor, order: int) -> torch.Tensor:
    """评估球谐基函数。

    Args:
        dirs: 归一化方向向量，形状 ``[..., 3]``（x, y, z）。
        order: 最高 SH 阶数（0, 1 或 2）。

    Returns:
        基函数值，形状 ``[..., (order+1)**2]``。
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}")

    x = dirs[..., 0:1]  # [..., 1]
    y = dirs[..., 1:2]
    z = dirs[..., 2:3]

    basis_parts: list[torch.Tensor] = []

    # Order 0
    basis_parts.append(torch.full_like(x, _C0))

    if order >= 1:
        # Order 1: 3 coeffs
        basis_parts.append(-_C1 * y)
        basis_parts.append(_C1 * z)
        basis_parts.append(-_C1 * x)

    if order >= 2:
        # Order 2: 5 coeffs
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z

        basis_parts.append(_C2[0] * xy)
        basis_parts.append(_C2[1] * yz)
        basis_parts.append(_C2[2] * (2.0 * zz - xx - yy))
        basis_parts.append(_C2[3] * xz)
        basis_parts.append(_C2[4] * (xx - yy))

    return torch.cat(basis_parts, dim=-1)


# ---------------------------------------------------------------------------
# decode_sh
# ---------------------------------------------------------------------------

def decode_sh(
    sh_texture: torch.Tensor,
    view_dirs: torch.Tensor,
    order: int = 2,
) -> torch.Tensor:
    """从 SH 纹理 + 视角方向解码颜色。

    Args:
        sh_texture: SH 系数，形状 ``[..., 27]``（9 基函数 × 3 通道）。
        view_dirs: 归一化方向向量，形状 ``[..., 3]``。
        order: 使用的 SH 阶数。

    Returns:
        解码颜色，形状 ``[..., 3]``。
    """
    n_sh = (order + 1) ** 2  # 使用多少个基函数

    # 评估基函数 → [..., n_sh]
    basis = eval_sh_basis(view_dirs, order)

    # 将 SH 纹理 reshape 为 [..., 9, 3]
    sh_reshaped = sh_texture.reshape(*sh_texture.shape[:-1], 9, 3)

    # 只取前 n_sh 个基函数的系数
    sh_used = sh_reshaped[..., :n_sh, :]  # [..., n_sh, 3]

    # basis[..., i] * sh[..., i, :] 对 i 求和
    # basis: [..., n_sh] → [..., n_sh, 1]
    basis_expanded = basis.unsqueeze(-1)  # [..., n_sh, 1]

    color = (sh_used * basis_expanded).sum(dim=-2)  # [..., 3]

    return color


# ---------------------------------------------------------------------------
# init_sh_texture
# ---------------------------------------------------------------------------

def init_sh_texture(
    resolution: int,
    sh_order: int = 2,
    init_dc: float = 0.5,
) -> nn.Parameter:
    """初始化 SH 纹理参数。

    Args:
        resolution: 纹理高度/宽度。
        sh_order: SH 阶数（最大 2）。
        init_dc: DC 分量初始颜色值（三通道均相同）。

    Returns:
        ``nn.Parameter``，形状 ``[1, H, W, 27]``。
    """
    n_coeffs = (sh_order + 1) ** 2  # 最多 9
    total = n_coeffs * 3             # 最多 27

    data = torch.zeros(1, resolution, resolution, total)

    # DC 系数 = init_dc / _C0（对 R, G, B 三个通道）
    dc_val = init_dc / _C0
    data[0, :, :, 0] = dc_val  # R channel DC
    data[0, :, :, 1] = dc_val  # G channel DC
    data[0, :, :, 2] = dc_val  # B channel DC

    return nn.Parameter(data)
