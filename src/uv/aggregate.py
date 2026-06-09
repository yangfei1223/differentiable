"""Per-triangle 渲染误差聚合。"""
from __future__ import annotations

import torch


def per_triangle_render_loss(
    pixel_loss: torch.Tensor,
    tri_ids: torch.Tensor,
    mask: torch.Tensor,
    num_faces: int,
) -> torch.Tensor:
    """将 per-pixel 渲染误差聚合为 per-triangle 平均误差。

    Args:
        pixel_loss: 每像素误差 [B, H, W, C]。
        tri_ids: 每像素所属三角形 ID [B, H, W]，int64。
            nvdiffrast normalizes triangle IDs to [0, 1], so this is tri_id_raw / num_faces.
        mask: 有效像素掩码 [B, H, W]，bool。
        num_faces: 总面数 F。

    Returns:
        每三角形平均误差 [F]。
    """
    scalar_loss = pixel_loss.mean(dim=-1)  # [B, H, W]

    flat_loss = scalar_loss[mask]
    flat_tri = tri_ids[mask]

    # nvdiffrast rast channel 3 is 1-based triangle ID (integer)
    flat_tri_idx = (flat_tri - 1).clamp(0, num_faces - 1).long()

    tri_loss_sum = torch.zeros(num_faces, device=pixel_loss.device, dtype=pixel_loss.dtype)
    tri_count = torch.zeros(num_faces, device=pixel_loss.device, dtype=pixel_loss.dtype)

    tri_loss_sum.scatter_add_(0, flat_tri_idx, flat_loss)
    tri_count.scatter_add_(0, flat_tri_idx, torch.ones_like(flat_loss))

    tri_mean = torch.where(tri_count > 0, tri_loss_sum / tri_count, torch.zeros_like(tri_loss_sum))

    return tri_mean
