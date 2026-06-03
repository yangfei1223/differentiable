"""损失函数 — L1、SSIM、Total Variation 及组合损失。

提供可微的损失函数用于可微烘焙优化：
- ``l1_loss``: 逐像素平均绝对误差
- ``ssim_loss``: 1 − SSIM（结构相似性）
- ``tv_loss``: 全变分正则化
- ``CombinedLoss``: 组合上述三种损失的 ``nn.Module``
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L1 Loss
# ---------------------------------------------------------------------------

def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐像素平均绝对误差。

    Args:
        pred: 预测图像，形状任意。
        target: 目标图像，与 pred 同形。

    Returns:
        标量损失。
    """
    return (pred - target).abs().mean()


# ---------------------------------------------------------------------------
# SSIM helpers
# ---------------------------------------------------------------------------

def _gaussian_window(
    window_size: int,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    """创建用于 SSIM 计算的高斯卷积核。

    Args:
        window_size: 窗口大小（奇数）。
        channels: 输入通道数。
        device: 设备。

    Returns:
        形状 ``[channels, 1, window_size, window_size]`` 的深度可分离卷积核。
    """
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # 外积 → [window_size, window_size]
    window_2d = g.unsqueeze(1) * g.unsqueeze(0)
    # 深度可分离 → [channels, 1, window_size, window_size]
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """1 − SSIM 损失。

    Args:
        pred: 预测图像，形状 ``[B, 3, H, W]``。
        target: 目标图像，与 pred 同形。
        window_size: 高斯窗口大小（默认 11）。

    Returns:
        标量损失（1 − SSIM）。
    """
    C = pred.shape[1]
    device = pred.device

    window = _gaussian_window(window_size, C, device)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
    mu_target = F.conv2d(target, window, padding=pad, groups=C)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_cross

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Total Variation Loss
# ---------------------------------------------------------------------------

def tv_loss(texture: torch.Tensor) -> torch.Tensor:
    """全变分（Total Variation）损失。

    Args:
        texture: 纹理图，形状 ``[1, H, W, C]``。

    Returns:
        标量损失。 (diff_h^2).mean() + (diff_w^2).mean()
    """
    # 沿高度方向差分: row[i+1] - row[i]
    diff_h = texture[:, 1:, :, :] - texture[:, :-1, :, :]
    # 沿宽度方向差分: col[j+1] - col[j]
    diff_w = texture[:, :, 1:, :] - texture[:, :, :-1, :]

    return (diff_h ** 2).mean() + (diff_w ** 2).mean()


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """组合损失：L1 + SSIM + TV。

    Args:
        lambda_l1: L1 损失权重。
        lambda_ssim: SSIM 损失权重。
        lambda_tv: TV 损失权重。
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_tv: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_tv = lambda_tv

    def forward(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        sh_texture: torch.Tensor,
    ) -> torch.Tensor:
        """前向计算组合损失。

        Args:
            rendered: 渲染图像，形状 ``[B, H, W, 3]``。
            gt: 真值图像，形状 ``[B, H, W, 3]``。
            mask: 有效区域掩码，形状 ``[B, H, W]``。
            sh_texture: SH 纹理参数，形状 ``[1, Ht, Wt, 27]``。

        Returns:
            标量组合损失。
        """
        # --- L1: masked pixel-wise ---
        # mask: [B, H, W] → [B, H, W, 1] 以广播到 3 通道
        mask_f = mask.unsqueeze(-1).float()  # [B, H, W, 1]
        abs_diff = (rendered - gt).abs() * mask_f  # [B, H, W, 3]
        # 除以 mask 总像素数 × 3
        n_valid = mask.sum() * 3 + 1e-8
        l1 = abs_diff.sum() / n_valid

        # --- SSIM: permute [B, H, W, 3] → [B, 3, H, W] ---
        rendered_chw = rendered.permute(0, 3, 1, 2)
        gt_chw = gt.permute(0, 3, 1, 2)
        ssim = ssim_loss(rendered_chw, gt_chw)

        # --- TV ---
        tv = tv_loss(sh_texture)

        return self.lambda_l1 * l1 + self.lambda_ssim * ssim + self.lambda_tv * tv
