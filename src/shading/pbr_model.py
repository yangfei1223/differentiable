"""PBR 着色模型 — Split-Sum 近似。"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.shading.base import ShadingModel
from src.shading.pbr.material import init_material_texture, decode_material, compute_F0
from src.shading.pbr.env_map import EnvironmentMap
from src.shading.pbr.brdf_lut import generate_brdf_lut, sample_brdf


class PBRShadingModel(ShadingModel):
    """PBR Split-Sum 着色模型。"""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mat_texture: nn.Parameter | None = None
        self.mat_textures: dict[str, nn.Parameter] = {}
        self.is_multi: bool = False
        self.env_map: EnvironmentMap | None = None
        self.brdf_lut: torch.Tensor | None = None

        pbr_cfg = config.pbr
        self.brdf_lut = generate_brdf_lut(pbr_cfg.brdf_lut_size)

    def parameters(self) -> list[nn.Parameter]:
        if self.is_multi:
            return list(self.mat_textures.values()) + [self.env_map.raw]
        return [self.mat_texture, self.env_map.raw]

    def init_textures(self, resolution: int, submesh_names: list[str] | None = None) -> None:
        pbr_cfg = self.config.pbr
        eh, ew = pbr_cfg.env_map_res

        if submesh_names is not None:
            self.is_multi = True
            self.mat_textures = {}
            for name in submesh_names:
                self.mat_textures[name] = nn.Parameter(
                    init_material_texture(resolution).data.to(self.device)
                )
        else:
            self.is_multi = False
            self.mat_texture = nn.Parameter(init_material_texture(resolution).data.to(self.device))

        self.env_map = EnvironmentMap(height=eh, width=ew).to(self.device)

    def shade(
        self,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
        tangents: torch.Tensor | None = None,
        bitangents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PBR split-sum 着色。"""
        import nvdiffrast.torch as dr

        # 1. 采样材质贴图 (单次 lookup: 8 通道 = 材质 + 法线)
        mat_raw = dr.texture(
            self.mat_texture, texc, filter_mode="linear", boundary_mode="clamp"
        )
        base_color, roughness, metallic, tex_normal = decode_material(mat_raw)
        # tex_normal: [1, H, W, 3] tangent-space 单位向量，(0,0,1)=无扰动
        if tangents is not None and bitangents is not None:
            world_normal = (
                tangents * tex_normal[..., 0:1] +
                bitangents * tex_normal[..., 1:2] +
                normals * tex_normal[..., 2:3]
            )
            normals = F.normalize(world_normal, dim=-1)

        # 3. 计算反射方向
        NdotV = (normals * view_dirs).sum(dim=-1, keepdim=True).clamp(0, 1)
        reflect_dir = 2.0 * NdotV * normals - view_dirs
        reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # 3. Diffuse 项
        irradiance = self.env_map.sample_diffuse(normals)
        F0 = compute_F0(base_color, metallic)
        kd = (1.0 - metallic) * (1.0 - F0)
        diffuse = kd * base_color * irradiance

        # 4. Specular 项
        prefiltered_color = self.env_map.sample_specular(reflect_dir, roughness)
        NdotV_flat = NdotV.reshape(-1)
        roughness_flat = roughness.reshape(-1)
        scale, bias = sample_brdf(self.brdf_lut.to(self.device), NdotV_flat, roughness_flat)
        scale = scale.reshape(*NdotV.shape)
        bias = bias.reshape(*NdotV.shape)
        specular = (F0 * scale + bias) * prefiltered_color

        # 5. 合成
        rgb = diffuse + specular
        rgb = rgb.clamp(0.0, 1.0)

        # 6. 遮罩
        mask = (rast_out[..., 3] > 0).float()
        rgb = rgb * mask.unsqueeze(-1)

        # 保存调试信息
        self._last_debug = {
            "diffuse": diffuse.detach(),
            "specular": specular.detach(),
            "base_color": base_color.detach(),
            "roughness": roughness.detach(),
            "metallic": metallic.detach(),
            "normal": normals.detach(),
        }

        return rgb, mask

    def shade_submesh(
        self,
        name: str,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
        tangents: torch.Tensor | None = None,
        bitangents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shade a specific submesh using its named texture."""
        import nvdiffrast.torch as dr

        tex = self.mat_textures[name]

        # 1. Sample material (single lookup: 8ch contains both material + normal)
        mat_raw = dr.texture(tex, texc, filter_mode="linear", boundary_mode="clamp")
        base_color, roughness, metallic, tex_normal = decode_material(mat_raw)
        if tangents is not None and bitangents is not None:
            world_normal = (
                tangents * tex_normal[..., 0:1] +
                bitangents * tex_normal[..., 1:2] +
                normals * tex_normal[..., 2:3]
            )
            normals = F.normalize(world_normal, dim=-1)

        # 3. Reflect direction
        NdotV = (normals * view_dirs).sum(dim=-1, keepdim=True).clamp(0, 1)
        reflect_dir = 2.0 * NdotV * normals - view_dirs
        reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # 4. Diffuse
        irradiance = self.env_map.sample_diffuse(normals)
        F0 = compute_F0(base_color, metallic)
        kd = (1.0 - metallic) * (1.0 - F0)
        diffuse = kd * base_color * irradiance

        # 5. Specular
        prefiltered_color = self.env_map.sample_specular(reflect_dir, roughness)
        NdotV_flat = NdotV.reshape(-1)
        roughness_flat = roughness.reshape(-1)
        scale, bias = sample_brdf(self.brdf_lut.to(self.device), NdotV_flat, roughness_flat)
        scale = scale.reshape(*NdotV.shape)
        bias = bias.reshape(*NdotV.shape)
        specular = (F0 * scale + bias) * prefiltered_color

        # 6. Combine
        rgb = diffuse + specular
        rgb = rgb.clamp(0.0, 1.0)

        # 7. Mask
        mask = (rast_out[..., 3] > 0).float()
        rgb = rgb * mask.unsqueeze(-1)

        self._last_debug = {
            "diffuse": diffuse.detach(),
            "specular": specular.detach(),
            "base_color": base_color.detach(),
            "roughness": roughness.detach(),
            "metallic": metallic.detach(),
            "normal": normals.detach(),
        }
        return rgb, mask

    def get_material_texture(self) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.is_multi:
            return {k: v.data.detach().cpu() for k, v in self.mat_textures.items()}
        return self.mat_texture.data.detach().cpu()

    def set_material_texture(self, texture: torch.Tensor | dict[str, torch.Tensor]) -> None:
        if isinstance(texture, dict):
            self.is_multi = True
            self.mat_textures = {
                k: nn.Parameter(v.to(self.device).contiguous())
                for k, v in texture.items()
            }
        else:
            self.is_multi = False
            self.mat_texture = nn.Parameter(texture.to(self.device).contiguous())

    def get_debug_info(self) -> dict:
        return getattr(self, "_last_debug", {})

    def state_dict(self) -> dict:
        if self.is_multi:
            return {
                "render_mode": "pbr",
                "is_multi": True,
                "mat_textures": {k: v.data.detach().cpu() for k, v in self.mat_textures.items()},
                "env_map": self.env_map.raw.data.detach().cpu(),
            }
        return {
            "render_mode": "pbr",
            "is_multi": False,
            "mat_texture": self.mat_texture.data.detach().cpu(),
            "env_map": self.env_map.raw.data.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("is_multi"):
            self.is_multi = True
            if "mat_textures" in state:
                self.mat_textures = {
                    k: nn.Parameter(v.to(self.device))
                    for k, v in state["mat_textures"].items()
                }
        else:
            self.is_multi = False
            if "mat_texture" in state:
                self.mat_texture = nn.Parameter(state["mat_texture"].to(self.device))
        if "env_map" in state:
            self.env_map.raw = nn.Parameter(state["env_map"].to(self.device))
        if "brdf_lut" in state:
            self.brdf_lut = state["brdf_lut"]

    def export(self, output_dir: str) -> list[str]:
        if self.is_multi:
            return self._export_multi(output_dir)
        return self._export_single(output_dir)

    def _export_single(self, output_dir: str) -> list[str]:
        import numpy as np
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        paths = []

        base_color, roughness, metallic, tex_normal = decode_material(self.mat_texture)

        # base_color.png
        bc = base_color[0].clamp(0, 1).pow(1.0 / 2.2).detach().cpu().numpy()
        bc = (bc * 255).astype(np.uint8)
        p = os.path.join(output_dir, "base_color.png")
        Image.fromarray(bc, "RGB").save(p)
        paths.append(p)

        # roughness.png
        r = roughness[0].clamp(0, 1).detach().cpu().numpy().repeat(3, axis=-1)
        r = (r * 255).astype(np.uint8)
        p = os.path.join(output_dir, "roughness.png")
        Image.fromarray(r, "RGB").save(p)
        paths.append(p)

        # metallic.png
        m = metallic[0].clamp(0, 1).detach().cpu().numpy().repeat(3, axis=-1)
        m = (m * 255).astype(np.uint8)
        p = os.path.join(output_dir, "metallic.png")
        Image.fromarray(m, "RGB").save(p)
        paths.append(p)

        # env_map.png
        p = os.path.join(output_dir, "env_map.png")
        self.env_map.export_image(p)
        paths.append(p)

        # normal_map.png — [-1,1] → [0,255]
        n_img = tex_normal[0].detach().cpu().numpy()  # [H, W, 3], 值域 [-1, 1]
        n_img = ((n_img + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        p = os.path.join(output_dir, "normal_map.png")
        Image.fromarray(n_img, "RGB").save(p)
        paths.append(p)

        return paths

    def _export_multi(self, output_dir: str) -> list[str]:
        import numpy as np
        from PIL import Image
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for name, tex in self.mat_textures.items():
            sub_dir = os.path.join(output_dir, name)
            os.makedirs(sub_dir, exist_ok=True)
            base_color, roughness, metallic, tex_normal = decode_material(tex)
            # base_color
            bc = base_color[0].clamp(0, 1).pow(1.0 / 2.2).detach().cpu().numpy()
            bc = (bc * 255).astype(np.uint8)
            p = os.path.join(sub_dir, "base_color.png")
            Image.fromarray(bc, "RGB").save(p); paths.append(p)
            # roughness
            r = roughness[0].clamp(0, 1).detach().cpu().numpy().repeat(3, axis=-1)
            r = (r * 255).astype(np.uint8)
            p = os.path.join(sub_dir, "roughness.png")
            Image.fromarray(r, "RGB").save(p); paths.append(p)
            # metallic
            m = metallic[0].clamp(0, 1).detach().cpu().numpy().repeat(3, axis=-1)
            m = (m * 255).astype(np.uint8)
            p = os.path.join(sub_dir, "metallic.png")
            Image.fromarray(m, "RGB").save(p); paths.append(p)
            # normal
            n_img = tex_normal[0].detach().cpu().numpy()
            n_img = ((n_img + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            p = os.path.join(sub_dir, "normal_map.png")
            Image.fromarray(n_img, "RGB").save(p); paths.append(p)
        # env_map
        p = os.path.join(output_dir, "env_map.png")
        self.env_map.export_image(p); paths.append(p)
        return paths
