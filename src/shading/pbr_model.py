"""PBR 着色模型 — Split-Sum 近似。"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.shading.base import ShadingModel
from src.shading.pbr.material import init_material_texture, decode_material, compute_F0
from src.shading.pbr.env_map import (
    init_env_map,
    prefilter_env_map,
    sample_prefiltered,
)
from src.shading.pbr.brdf_lut import generate_brdf_lut, sample_brdf


class PBRShadingModel(ShadingModel):
    """PBR Split-Sum 着色模型。"""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mat_texture: nn.Parameter | None = None
        self.env_map: nn.Parameter | None = None
        self.brdf_lut: torch.Tensor | None = None

        pbr_cfg = config.pbr
        self.brdf_lut = generate_brdf_lut(pbr_cfg.brdf_lut_size)
        self.n_mip_levels = pbr_cfg.n_mip_levels

    def parameters(self) -> list[nn.Parameter]:
        return [self.mat_texture, self.env_map]

    def init_textures(self, resolution: int) -> None:
        pbr_cfg = self.config.pbr
        eh, ew = pbr_cfg.env_map_res

        self.mat_texture = init_material_texture(resolution).to(self.device)
        self.env_map = init_env_map(eh, ew).to(self.device)

    def shade(
        self,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PBR split-sum 着色。"""
        import nvdiffrast.torch as dr

        # 1. 采样材质贴图
        mat_raw = dr.texture(
            self.mat_texture, texc, filter_mode="linear", boundary_mode="clamp"
        )
        base_color, roughness, metallic = decode_material(mat_raw)

        # 2. 计算反射方向
        NdotV = (normals * view_dirs).sum(dim=-1, keepdim=True).clamp(0, 1)
        reflect_dir = 2.0 * NdotV * normals - view_dirs
        reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # 3. 预滤波环境贴图
        prefiltered = prefilter_env_map(self.env_map, self.n_mip_levels)

        # 4. Diffuse 项
        irradiance = sample_prefiltered(prefiltered, normals, torch.zeros_like(NdotV), self.n_mip_levels)
        F0 = compute_F0(base_color, metallic)
        kd = (1.0 - metallic) * (1.0 - F0)
        diffuse = kd * base_color * irradiance

        # 5. Specular 项
        prefiltered_color = sample_prefiltered(prefiltered, reflect_dir, roughness, self.n_mip_levels)
        NdotV_flat = NdotV.reshape(-1)
        roughness_flat = roughness.reshape(-1)
        scale, bias = sample_brdf(self.brdf_lut, NdotV_flat, roughness_flat)
        scale = scale.reshape(*NdotV.shape)
        bias = bias.reshape(*NdotV.shape)
        specular = (F0 * scale + bias) * prefiltered_color

        # 6. 合成
        rgb = diffuse + specular
        rgb = rgb.clamp(0.0, 1.0)

        # 7. 遮罩
        mask = (rast_out[..., 3] > 0).float()
        rgb = rgb * mask.unsqueeze(-1)

        # 保存调试信息
        self._last_debug = {
            "diffuse": diffuse.detach(),
            "specular": specular.detach(),
            "base_color": base_color.detach(),
            "roughness": roughness.detach(),
            "metallic": metallic.detach(),
        }

        return rgb, mask

    def get_material_texture(self) -> torch.Tensor:
        return self.mat_texture.data.detach().cpu()

    def set_material_texture(self, texture: torch.Tensor) -> None:
        self.mat_texture = nn.Parameter(texture.to(self.device).contiguous())

    def get_debug_info(self) -> dict:
        return getattr(self, "_last_debug", {})

    def state_dict(self) -> dict:
        return {
            "render_mode": "pbr",
            "mat_texture": self.mat_texture.data.detach().cpu(),
            "env_map": self.env_map.data.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if "mat_texture" in state:
            self.mat_texture = nn.Parameter(state["mat_texture"].to(self.device))
        if "env_map" in state:
            self.env_map = nn.Parameter(state["env_map"].to(self.device))

    def export(self, output_dir: str) -> list[str]:
        import numpy as np
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        paths = []

        base_color, roughness, metallic = decode_material(self.mat_texture)

        # base_color.png
        bc = base_color[0].clamp(0, 1).pow(1.0 / 2.2).cpu().numpy()
        bc = (bc * 255).astype(np.uint8)
        p = os.path.join(output_dir, "base_color.png")
        Image.fromarray(bc, "RGB").save(p)
        paths.append(p)

        # roughness.png
        r = roughness[0].clamp(0, 1).cpu().numpy().repeat(3, axis=-1)
        r = (r * 255).astype(np.uint8)
        p = os.path.join(output_dir, "roughness.png")
        Image.fromarray(r, "RGB").save(p)
        paths.append(p)

        # metallic.png
        m = metallic[0].clamp(0, 1).cpu().numpy().repeat(3, axis=-1)
        m = (m * 255).astype(np.uint8)
        p = os.path.join(output_dir, "metallic.png")
        Image.fromarray(m, "RGB").save(p)
        paths.append(p)

        # env_map.png
        from src.shading.pbr.env_map import _decode_env_map
        env_decoded = _decode_env_map(self.env_map)
        env_img = env_decoded[0].clamp(0, 1).cpu().numpy()
        env_img = (env_img * 255).astype(np.uint8)
        p = os.path.join(output_dir, "env_map.png")
        Image.fromarray(env_img, "RGB").save(p)
        paths.append(p)

        return paths
