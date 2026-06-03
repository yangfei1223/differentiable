"""UV seam padding — 边界膨胀算子。

将纹理图集中 UV seam 处的空白区域用邻近有效像素的颜色均值填充，
避免渲染时在边界处产生黑色伪影。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate_texture(
    texture: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int = 3,
) -> torch.Tensor:
    """用邻域均值膨胀纹理的空白区域。

    Parameters
    ----------
    texture : Tensor [1, H, W, C]
        输入纹理图集（含空白区域）。
    valid_mask : Tensor [1, H, W, 1]
        有效像素掩码，1=有效，0=空白。
    radius : int
        膨胀半径（kernel size = 2*radius+1）。

    Returns
    -------
    Tensor [1, H, W, C]
        空白区域被邻域颜色均值填充后的纹理；原始有效像素保持不变。
    """
    _, H, W, C = texture.shape
    k = 2 * radius + 1

    # 构建 uniform kernel [k, k]
    kernel = torch.ones(1, 1, k, k, dtype=texture.dtype, device=texture.device)
    kernel = kernel / (k * k)

    # 准备 NCHW 格式用于 conv2d
    # texture: [1,H,W,C] → [C,1,H,W]
    tex_nchw = texture.permute(0, 3, 1, 2).squeeze(0)  # [C, H, W]
    mask_nchw = valid_mask.permute(0, 3, 1, 2)          # [1, 1, H, W]

    # 对每个通道独立做 conv2d：tex_ch * mask 与 mask 分别卷积
    # tex_masked: [C, 1, H, W]
    tex_masked = tex_nchw.unsqueeze(1) * mask_nchw  # broadcast: [C,1,H,W]

    # 加权求和 (numerator): conv2d(tex * mask, kernel)  → [C, 1, H, W]
    # 权重求和 (denominator): conv2d(mask, kernel)       → [1, 1, H, W]
    pad = radius
    numerator = F.conv2d(
        tex_masked.reshape(1, C, H, W),          # [1, C, H, W]
        kernel.expand(C, 1, k, k),               # [C, 1, k, k] — depthwise-like
        padding=pad,
        groups=C,
    )  # [1, C, H, W]

    denominator = F.conv2d(mask_nchw, kernel, padding=pad)  # [1, 1, H, W]

    # 邻域均值 = numerator / denominator（避免除零）
    safe_denom = denominator.clamp(min=1e-8)
    avg_color = (numerator / safe_denom)  # [1, C, H, W]

    # 转回 [1, H, W, C]
    avg_color = avg_color.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]

    # 只填充空白区域 (mask == 0)；有效区域保留原值
    inv_mask = 1.0 - valid_mask
    result = texture * valid_mask + avg_color * inv_mask

    return result
