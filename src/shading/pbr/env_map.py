"""Equirectangular 环境贴图 — nn.Module，nvdiffrast 采样。"""
from __future__ import annotations

import math

import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentMap(nn.Module):
    """可训练的 HDR 环境贴图。

    参数存储为 raw 值，经 softplus 解码为 HDR ≥ 0。
    mipmap 由 nvdiffrast 内置 box filter 自动生成。

    Args:
        height: 贴图高度（建议 256）。
        width: 贴图宽度（建议 512，2:1 equirect）。
        init_image: 可选的初始图像 [1, H, W, 3]（HDR 值域 ≥ 0）。
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 512,
        init_image: torch.Tensor | None = None,
    ):
        super().__init__()
        self.height = height
        self.width = width

        if init_image is not None:
            img = init_image.clone().float()
            if img.dim() == 3:
                img = img.unsqueeze(0)
            data = torch.log(torch.exp(img.clamp(min=1e-6)) - 1.0)
        else:
            # softplus_inv(0.5) = log(exp(0.5)-1) ≈ -0.193
            init_val = math.log(math.exp(0.5) - 1.0 + 1e-6)
            data = torch.full((1, height, width, 3), init_val)

        self.raw = nn.Parameter(data)

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(self) -> torch.Tensor:
        """解码 raw 参数 → HDR 非负值 [1, H, W, 3]。"""
        return F.softplus(self.raw)

    # ------------------------------------------------------------------
    # 方向 → UV
    # ------------------------------------------------------------------
    @staticmethod
    def direction_to_uv(dirs: torch.Tensor) -> torch.Tensor:
        """方向向量 → equirect UV 坐标 [*, 2]，值域 [0, 1]。
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        u = torch.atan2(z, x) / (2.0 * math.pi) + 0.5
        v = torch.asin(y.clamp(-0.999, 0.999)) / math.pi + 0.5
        return torch.stack([u, v], dim=-1)

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def sample(
        self,
        dirs: torch.Tensor,
        mip_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """沿方向采样环境贴图。

        Args:
            dirs: 方向向量 [*, 3]。
            mip_level: 每 pixel 的 mip level [*, 1]。
                None = level 0（最清晰）。

        Returns:
            颜色 [*, 3]。
        """
        decoded = self.decode()  # [1, H, W, 3]
        H, W = decoded.shape[1], decoded.shape[2]
        max_mip = int(math.floor(math.log2(max(H, W))))

        uv = self.direction_to_uv(dirs)
        orig_shape = uv.shape[:-1]
        grid = uv.reshape(1, 1, -1, 2)

        kwargs: dict = dict(
            filter_mode="linear",
            boundary_mode="wrap",
        )
        if mip_level is not None:
            kwargs["filter_mode"] = "linear-mipmap-linear"
            kwargs["max_mip_level"] = max_mip
            kwargs["mip_level_bias"] = mip_level.reshape(1, 1, -1)

        color = dr.texture(decoded, grid, **kwargs)  # [1, 1, N, 3]
        return color.reshape(*orig_shape, 3)

    def sample_diffuse(self, normal: torch.Tensor) -> torch.Tensor:
        """采样 diffuse irradiance（法线方向，最模糊 mip level）。

        Args:
            normal: 法线 [*, 3]。
        Returns:
            irradiance [*, 3]。
        """
        H, W = self.height, self.width
        max_mip = int(math.floor(math.log2(max(H, W))))
        mip = torch.full(normal.shape[:-1] + (1,), float(max_mip),
                         device=normal.device, dtype=normal.dtype)
        return self.sample(normal, mip_level=mip)

    def sample_specular(
        self,
        reflect_dir: torch.Tensor,
        roughness: torch.Tensor,
    ) -> torch.Tensor:
        """采样 specular 预滤波颜色。

        Args:
            reflect_dir: 反射方向 [*, 3]。
            roughness: 粗糙度 [*, 1]，值域 [0, 1]。
        Returns:
            预滤波颜色 [*, 3]。
        """
        H, W = self.height, self.width
        max_mip = int(math.floor(math.log2(max(H, W))))
        mip = roughness * max_mip
        return self.sample(reflect_dir, mip_level=mip)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export_image(self, output_path: str) -> str:
        """导出解码后的 env map 为 PNG。"""
        from PIL import Image
        import numpy as np
        decoded = self.decode()
        img = decoded[0].clamp(0, 1).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img, "RGB").save(output_path)
        return output_path
