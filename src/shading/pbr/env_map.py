"""Equirectangular 环境贴图 — nn.Module 封装，高斯预滤波 mipmap + 方向采样。"""
from __future__ import annotations

import math

import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentMap(nn.Module):
    """可训练的 HDR 环境贴图。

    参数存储为 raw 值，经 softplus 解码为 HDR ≥ 0。
    支持高斯预滤波 mipmap（每步前向实时计算，梯度回传）和方向采样。

    Args:
        height: 贴图高度（建议 256）。
        width: 贴图宽度（建议 512，2:1 equirect）。
        n_mip_levels: mipmap 级别数（包含 level 0）。
            None 则自动按分辨率计算 floor(log2(max(H,W)))+1。
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 512,
        n_mip_levels: int | None = None,
        init_image: torch.Tensor | None = None,
    ):
        super().__init__()
        self.height = height
        self.width = width

        if n_mip_levels is not None:
            self.n_mip_levels = n_mip_levels
        else:
            self.n_mip_levels = int(math.floor(math.log2(max(height, width)))) + 1

        if init_image is not None:
            # 传入图像 [1, H, W, 3] 或 [H, W, 3]，值域 ≥ 0
            img = init_image.clone().float()
            if img.dim() == 3:
                img = img.unsqueeze(0)
            assert img.shape == (1, height, width, 3), (
                f"init_image shape {img.shape} != expected (1, {height}, {width}, 3)"
            )
            # softplus_inv(x) = log(exp(x) - 1)，x > 0
            data = torch.log(torch.exp(img.clamp(min=1e-6)) - 1.0)
        else:
            # softplus_inv(0.5) = log(exp(0.5)-1) ≈ -0.193，初始解码值 ≈ 0.5
            init_val = math.log(math.exp(0.5) - 1.0 + 1e-6)
            data = torch.full((1, height, width, 3), init_val)

        self.raw = nn.Parameter(data)

        # 高斯卷积核缓存: {(kernel_size, sigma, device): kernel_3ch}
        self._kernel_cache: dict[tuple, torch.Tensor] = {}

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

        Args:
            dirs: 归一化方向 [*, 3] (x, y, z)
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        u = torch.atan2(z, x) / (2.0 * math.pi) + 0.5
        v = torch.asin(y.clamp(-0.999, 0.999)) / math.pi + 0.5
        return torch.stack([u, v], dim=-1)

    # ------------------------------------------------------------------
    # 高斯预滤波 mipmap
    # ------------------------------------------------------------------
    def _get_gauss_kernel(self, sigma: float, device: torch.device, max_size: int = 0) -> torch.Tensor:
        """获取/缓存高斯卷积核 [3, 1, k, k]（groups=3 卷积用）。

        Args:
            sigma: 高斯 sigma。
            device: 设备。
            max_size: 最大核尺寸限制（0=不限制）。
        """
        k = int(sigma * 4) | 1
        k = max(k, 3)
        if max_size > 0:
            k = min(k, max_size)
        if k % 2 == 0:
            k += 1
        cache_key = (k, sigma, device)
        if cache_key not in self._kernel_cache:
            ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kern = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-8))
            kern = kern / kern.sum()
            self._kernel_cache[cache_key] = (
                kern.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
            )
        return self._kernel_cache[cache_key]

    def build_mipmap(self) -> list[torch.Tensor]:
        """生成高斯预滤波 mipmap 链（每步前向调用，梯度流回 self.raw）。

        每级：水平环绕 padding → 高斯模糊 → 2× 下采样。
        sigma 随 level 线性递增。

        Returns:
            list of [1, H/2^L, W/2^L, 3]，从 level 1 到 level (n_mip_levels-1)。
            level 0 就是 self.decode()。
        """
        decoded = self.decode()  # [1, H, W, 3]
        current = decoded[0].permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        device = current.device

        mip_chain: list[torch.Tensor] = []
        for level in range(1, self.n_mip_levels):
            roughness = level / max(self.n_mip_levels - 1, 1)
            h_cur, w_cur = current.shape[2], current.shape[3]
            sigma = max(roughness * min(h_cur, w_cur) * 0.25, 0.5)

            # 限制 kernel 不超过当前分辨率
            max_k = min(h_cur, w_cur)
            if max_k % 2 == 0:
                max_k -= 1
            max_k = max(max_k, 1)

            kernel = self._get_gauss_kernel(sigma, device, max_size=max_k)
            pad = kernel.shape[-1] // 2

            # 限制 pad 不超过当前分辨率的一半
            h_pad = min(pad, max((h_cur - 1) // 2, 0))
            w_pad = min(pad, max((w_cur - 1) // 2, 0))

            # 水平环绕 padding：equirect 左右边缘是连续的
            padded = F.pad(current, (w_pad, w_pad, 0, 0), mode="circular")
            # 上下 reflect padding（极点不环绕，但避免 pad > 尺寸）
            if h_pad < h_cur:
                padded = F.pad(padded, (0, 0, h_pad, h_pad), mode="reflect")

            blurred = F.conv2d(padded, kernel, padding=0, groups=3)

            h_new = max(h_cur // 2, 1)
            w_new = max(w_cur // 2, 1)
            downsampled = F.interpolate(blurred, size=(h_new, w_new), mode="bilinear", align_corners=False)

            mip_chain.append(downsampled.squeeze(0).permute(1, 2, 0).unsqueeze(0).contiguous())
            current = downsampled

        return mip_chain

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
                值越大越模糊（高斯预滤波）。

        Returns:
            颜色 [*, 3]。
        """
        decoded = self.decode()  # [1, H, W, 3]
        uv = self.direction_to_uv(dirs)  # [*, 2]
        orig_shape = uv.shape[:-1]
        grid = uv.reshape(1, 1, -1, 2)  # [1, 1, N, 2]

        kwargs: dict = dict(
            filter_mode="linear",
            boundary_mode="wrap",
        )

        if mip_level is not None:
            mip_chain = self.build_mipmap()
            kwargs["filter_mode"] = "linear-mipmap-linear"
            kwargs["max_mip_level"] = self.n_mip_levels - 1
            kwargs["mip"] = mip_chain
            kwargs["mip_level_bias"] = mip_level.reshape(1, 1, -1)

        color = dr.texture(decoded, grid, **kwargs)  # [1, 1, N, 3]
        return color.reshape(*orig_shape, 3)

    def sample_diffuse(self, normals: torch.Tensor) -> torch.Tensor:
        """采样 diffuse irradiance（法线方向，最模糊 mip level）。

        Args:
            normals: 法线 [*, 3]。

        Returns:
            irradiance [*, 3]。
        """
        max_mip = torch.full(
            normals.shape[:-1] + (1,),
            float(self.n_mip_levels - 1),
            device=normals.device,
            dtype=normals.dtype,
        )
        return self.sample(normals, mip_level=max_mip)

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
        mip_level = roughness * (self.n_mip_levels - 1)
        return self.sample(reflect_dir, mip_level=mip_level)

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

    def state_dict_for_pbr(self) -> dict:
        """返回用于 PBR checkpoint 的 state dict 片段。"""
        return {"env_map_raw": self.raw.data.detach().cpu()}

    def load_state_dict_for_pbr(self, state: dict) -> None:
        """从 PBR checkpoint 恢复。"""
        key = "env_map_raw" if "env_map_raw" in state else "env_map"
        if key in state:
            self.raw = nn.Parameter(state[key].to(self.raw.device))
