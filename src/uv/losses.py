"""UV 正则化损失 — Symmetric Dirichlet + 面积保持。"""
from __future__ import annotations

import torch
import torch.nn as nn


def _triangle_uv_areas(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """计算每个三角形的 UV 面积。"""
    v0 = uv[faces[:, 0]]
    v1 = uv[faces[:, 1]]
    v2 = uv[faces[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    return cross.abs() * 0.5


class SymDirichletLoss(nn.Module):
    """Content-Aware Symmetric Dirichlet Energy。

    E_sym = σ1² + σ2² + 1/σ1² + 1/σ2²
    L = Σ_tri  E_sym(tri) × render_loss(tri)
    """

    def forward(
        self,
        uv: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        per_tri_render_loss: torch.Tensor,
    ) -> torch.Tensor:
        uv0 = uv[faces[:, 0]]
        uv1 = uv[faces[:, 1]]
        uv2 = uv[faces[:, 2]]

        du1 = uv1[:, 0] - uv0[:, 0]
        dv1 = uv1[:, 1] - uv0[:, 1]
        du2 = uv2[:, 0] - uv0[:, 0]
        dv2 = uv2[:, 1] - uv0[:, 1]

        p0 = vertices[faces[:, 0]]
        p1 = vertices[faces[:, 1]]
        p2 = vertices[faces[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0

        det = du1 * dv2 - du2 * dv1
        det_safe = torch.where(det.abs() < 1e-8, torch.ones_like(det) * 1e-8, det)
        inv_det = 1.0 / det_safe

        j_col0 = e1 * (dv2 * inv_det).unsqueeze(-1) + e2 * (-dv1 * inv_det).unsqueeze(-1)
        j_col1 = e1 * (-du2 * inv_det).unsqueeze(-1) + e2 * (du1 * inv_det).unsqueeze(-1)

        J = torch.stack([j_col0, j_col1], dim=-1)  # [F, 3, 2]

        JtJ = torch.bmm(J.transpose(1, 2), J)  # [F, 2, 2]
        trace = JtJ[:, 0, 0] + JtJ[:, 1, 1]
        det_JtJ = JtJ[:, 0, 0] * JtJ[:, 1, 1] - JtJ[:, 0, 1] * JtJ[:, 1, 0]

        det_JtJ_safe = torch.where(det_JtJ.abs() < 1e-10,
                                    torch.sign(det_JtJ) * 1e-10 + (det_JtJ.abs() < 1e-10).float() * 1e-10,
                                    det_JtJ)
        e_sym = trace + trace / det_JtJ_safe.abs()

        flip_penalty = torch.where(det < 0, (det.abs() + 1.0) * 10.0, torch.zeros_like(det))
        e_sym = e_sym + flip_penalty

        loss = (e_sym * per_tri_render_loss).mean()
        return loss


class AreaPreserveLoss(nn.Module):
    """面积保持正则化。 L = mean(|uv_area - target_area|)"""

    def forward(
        self,
        uv: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        target_areas: torch.Tensor,
    ) -> torch.Tensor:
        current_areas = _triangle_uv_areas(uv, faces)
        return (current_areas - target_areas).abs().mean()
