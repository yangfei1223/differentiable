"""可微渲染器 — 基于 nvdiffrast 的前向管线与 SH 解码。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.sh import eval_sh_basis


def _get_dr():
    """延迟导入 nvdiffrast — 仅在需要时才触发，避免无 CUDA 环境下导入失败。"""
    import nvdiffrast.torch as _dr
    return _dr


class DifferentiableRenderer:
    """基于 nvdiffrast 的可微渲染器。

    支持 SH 纹理（球谐系数纹理），在渲染时根据视角方向动态解码颜色。

    Args:
        vertices: 顶点位置，形状 ``[V, 3]`` 或 ``[1, V, 3]``。
        faces: 三角面索引，形状 ``[F, 3]``，int 类型。
        uvs: UV 坐标，形状 ``[Vt, 2]``。
        uv_idx: UV 索引，形状 ``[F, 3]``，int 类型。
        resolution: 默认渲染分辨率（高=宽）。
        device: 渲染设备。
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        uvs: torch.Tensor,
        uv_idx: torch.Tensor,
        resolution: int = 512,
        device: str = "cuda",
    ):
        dr = _get_dr()

        self.resolution = resolution
        self.device = device

        # 确保顶点形状为 [1, V, 3]
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        self.vertices = vertices.to(device).float()

        # faces: [F, 3] int32
        self.faces = faces.to(device).int()

        # uvs: [Vt, 2]
        self.uvs = uvs.to(device).float()

        # uv_idx: [F, 3] int32
        self.uv_idx = uv_idx.to(device).int()

        # nvdiffrast GL 上下文
        self.glctx = dr.RasterizeGLContext()

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def render(
        self,
        sh_texture: nn.Parameter,
        camera,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """渲染一帧。

        Args:
            sh_texture: SH 系数纹理，形状 ``[1, H_tex, W_tex, 27]``。
            camera: :class:`Camera` 对象（需提供 ``mvp_torch()`` 与
                ``position``）。

        Returns:
            ``(rgb, mask)`` —
            ``rgb`` 形状 ``[1, H, W, 3]``，``mask`` 形状 ``[1, H, W]``。
        """
        dr = _get_dr()
        h = w = self.resolution

        # ---- 1. MVP 矩阵 ----
        mvp = camera.mvp_torch().to(self.device)  # [1, 4, 4]

        # ---- 2. 裁剪空间顶点 ----
        verts = self.vertices  # [1, V, 3]
        ones = torch.ones_like(verts[..., :1])  # [1, V, 1]
        verts_h = torch.cat([verts, ones], dim=-1)  # [1, V, 4]

        # (1,V,4) @ (1,4,4)^T → clip space
        clip = torch.bmm(verts_h, mvp.transpose(1, 2))  # [1, V, 4]

        # ---- 3. 光栅化 ----
        rast, _ = dr.rasterize(self.glctx, clip, self.faces, resolution=[h, w])

        # ---- 4. 插值 UV 坐标 ----
        texc, _ = dr.interpolate(self.uvs, rast, self.uv_idx)  # [1, H, W, 2]

        # ---- 5. 插值世界坐标 ----
        world_pos, _ = dr.interpolate(self.vertices, rast, self.faces)  # [1, H, W, 3]

        # ---- 6. 视角方向 ----
        cam_pos = (
            torch.tensor(camera.position, dtype=torch.float32, device=self.device)
            .reshape(1, 1, 1, 3)
        )
        view_dir = cam_pos - world_pos  # [1, H, W, 3]
        view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # ---- 7. SH 纹理采样 ----
        tex = dr.texture(
            sh_texture,
            texc,
            filter_mode="linear",
            boundary_mode="zero",
        )  # [1, H, W, 27]

        # ---- 8. SH 解码 ----
        # reshape → [1, H, W, 9, 3]
        sh_9x3 = tex.reshape(*tex.shape[:-1], 9, 3)

        # 评估 SH 基函数 → [1, H, W, 9]
        basis = eval_sh_basis(view_dir, order=2)  # [1, H, W, 9]

        # 加权求和 → [1, H, W, 3]
        basis_exp = basis.unsqueeze(-1)  # [1, H, W, 9, 1]
        rgb = (sh_9x3 * basis_exp).sum(dim=-2)  # [1, H, W, 3]

        # clamp 负值：SH 高阶系数可能产生负贡献，截断到 0
        rgb = rgb.clamp(min=0.0)

        # ---- 9. 遮罩 ----
        mask = (rast[..., 3] > 0).float()  # [1, H, W]

        # ---- 10. 应用遮罩 ----
        rgb = rgb * mask.unsqueeze(-1)  # [1, H, W, 3]

        return rgb, mask
