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
        normals: torch.Tensor = None,
        normal_idx: torch.Tensor = None,
        tangents: torch.Tensor = None,
        bitangents: torch.Tensor = None,
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

        if normals is not None:
            self.normals = normals.to(device).float()
        else:
            self.normals = None
        if normal_idx is not None:
            self.normal_idx = normal_idx.to(device).int()
        else:
            self.normal_idx = None

        if tangents is not None:
            self.tangents = tangents.to(device).float()
        else:
            self.tangents = None
        if bitangents is not None:
            self.bitangents = bitangents.to(device).float()
        else:
            self.bitangents = None

        # nvdiffrast GL 上下文
        self.glctx = dr.RasterizeGLContext()

    def set_uvs(self, uvs: torch.Tensor) -> None:
        """更新 UV 坐标（用于 UV 优化）。

        Args:
            uvs: UV 坐标 [1, V, 2] 或 [V, 2]，值在 (0, 1)。
        """
        if uvs.dim() == 2:
            uvs = uvs.unsqueeze(0)
        self.uvs = uvs.to(self.device).float()

    # ------------------------------------------------------------------
    # rasterize_and_interpolate
    # ------------------------------------------------------------------
    def rasterize_and_interpolate(
        self, camera
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """光栅化 + 插值，返回中间结果供着色模型使用。

        Returns:
            (rast_out, texc, world_pos, interp_normals, view_dirs, interp_tangents, interp_bitangents)
        """
        dr = _get_dr()
        h = w = self.resolution

        mvp = camera.mvp_torch().to(self.device)
        verts = self.vertices
        ones = torch.ones_like(verts[..., :1])
        verts_h = torch.cat([verts, ones], dim=-1)
        clip = torch.bmm(verts_h, mvp.transpose(1, 2))

        rast, _ = dr.rasterize(self.glctx, clip, self.faces, resolution=[h, w])
        texc, _ = dr.interpolate(self.uvs, rast, self.uv_idx)
        world_pos, _ = dr.interpolate(self.vertices, rast, self.faces)

        if self.normals is not None and self.normal_idx is not None:
            interp_normals, _ = dr.interpolate(self.normals, rast, self.normal_idx)
            interp_normals = interp_normals / (interp_normals.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            interp_normals = torch.zeros_like(world_pos)

        cam_pos = (
            torch.tensor(camera.position, dtype=torch.float32, device=self.device)
            .reshape(1, 1, 1, 3)
        )
        view_dirs = cam_pos - world_pos
        view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # 插值切线/副切线
        if self.tangents is not None and self.bitangents is not None:
            interp_tangents, _ = dr.interpolate(self.tangents, rast, self.faces)
            interp_tangents = interp_tangents / (interp_tangents.norm(dim=-1, keepdim=True) + 1e-8)
            interp_bitangents, _ = dr.interpolate(self.bitangents, rast, self.faces)
            interp_bitangents = interp_bitangents / (interp_bitangents.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            interp_tangents = torch.zeros_like(view_dirs)
            interp_bitangents = torch.zeros_like(view_dirs)

        return rast, texc, world_pos, interp_normals, view_dirs, interp_tangents, interp_bitangents

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def render(
        self,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        camera,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """渲染一帧。

        Args:
            features_dc: SH DC 系数纹理，形状 ``[1, H_tex, W_tex, 3]``。
            features_rest: SH 高阶系数纹理，形状 ``[1, H_tex, W_tex, (n-1)*3]``。
            camera: :class:`Camera` 对象（需提供 ``mvp_torch()`` 与
                ``position``）。

        Returns:
            ``(rgb, mask, interp_normals)`` —
            ``rgb`` 形状 ``[1, H, W, 3]``，``mask`` 形状 ``[1, H, W]``，
            ``interp_normals`` 形状 ``[1, H, W, 3]``。
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

        # ---- 5b. 插值法线 ----
        if self.normals is not None and self.normal_idx is not None:
            interp_normals, _ = dr.interpolate(self.normals, rast, self.normal_idx)
            interp_normals = interp_normals / (interp_normals.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            interp_normals = torch.zeros_like(world_pos)

        # ---- 6. 视角方向 ----
        cam_pos = (
            torch.tensor(camera.position, dtype=torch.float32, device=self.device)
            .reshape(1, 1, 1, 3)
        )
        view_dir = cam_pos - world_pos  # [1, H, W, 3]
        view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # ---- 7. 拼接 DC + Rest → 完整 SH 纹理并采样 ----
        full_tex = torch.cat([features_dc, features_rest], dim=-1)  # [1, H, W, n*3]
        tex = dr.texture(
            full_tex,
            texc,
            filter_mode="linear",
            boundary_mode="clamp",
        )  # [1, H, W, C]

        # ---- 8. SH 解码（3DGS 风格） ----
        n_sh = tex.shape[-1] // 3  # SH 系数个数: 1(order0), 4(order1), 9(order2)
        sh_order = int(n_sh ** 0.5) - 1
        sh_nx3 = tex.reshape(*tex.shape[:-1], n_sh, 3)

        # 评估 SH 基函数 → [1, H, W, n_sh]
        basis = eval_sh_basis(view_dir, order=sh_order)

        # 加权求和 → [1, H, W, 3]
        basis_exp = basis.unsqueeze(-1)  # [1, H, W, n_sh, 1]
        rgb = (sh_nx3 * basis_exp).sum(dim=-2)  # [1, H, W, 3]

        # 3DGS 约定: SH 输出 + 0.5 还原为 RGB，然后 clamp
        rgb = rgb + 0.5
        rgb = rgb.clamp(0.0, 1.0)

        # ---- 9. 遮罩 ----
        mask = (rast[..., 3] > 0).float()  # [1, H, W]

        # ---- 10. 应用遮罩 ----
        rgb = rgb * mask.unsqueeze(-1)  # [1, H, W, 3]

        return rgb, mask, interp_normals
