"""SH 着色模型包装 — 将现有 SH 管线封装为 ShadingModel 接口。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import Config
from src.sh import (
    init_sh_texture,
    cat_sh_features,
    eval_sh_basis,
)
from src.shading.base import ShadingModel


class SHShadingModel(ShadingModel):
    """SH 着色模型。"""

    def __init__(self, config: Config):
        self.config = config
        self.sh_order = config.texture.sh_order
        self.features_dc: nn.Parameter | None = None
        self.features_rest: nn.Parameter | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def parameters(self) -> list[nn.Parameter]:
        return [self.features_dc, self.features_rest]

    def init_textures(self, resolution: int) -> None:
        _dc, _rest = init_sh_texture(
            resolution,
            sh_order=self.sh_order,
            init_dc=self.config.texture.init_dc_value,
        )
        self.features_dc = nn.Parameter(_dc.data.to(self.device))
        self.features_rest = nn.Parameter(_rest.data.to(self.device))

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
        import nvdiffrast.torch as dr

        full_tex = torch.cat([self.features_dc, self.features_rest], dim=-1)
        tex = dr.texture(full_tex, texc, filter_mode="linear", boundary_mode="clamp")

        n_sh = tex.shape[-1] // 3
        sh_order = int(n_sh ** 0.5) - 1
        sh_nx3 = tex.reshape(*tex.shape[:-1], n_sh, 3)
        basis = eval_sh_basis(view_dirs, order=sh_order)
        basis_exp = basis.unsqueeze(-1)
        rgb = (sh_nx3 * basis_exp).sum(dim=-2)
        rgb = rgb + 0.5
        rgb = rgb.clamp(0.0, 1.0)

        mask = (rast_out[..., 3] > 0).float()
        rgb = rgb * mask.unsqueeze(-1)

        return rgb, mask

    def get_material_texture(self) -> torch.Tensor:
        return cat_sh_features(self.features_dc, self.features_rest).detach().cpu()

    def set_material_texture(self, texture: torch.Tensor) -> None:
        self.features_dc = nn.Parameter(texture[..., :3].to(self.device).contiguous())
        self.features_rest = nn.Parameter(texture[..., 3:].to(self.device).contiguous())

    def state_dict(self) -> dict:
        return {
            "render_mode": "sh",
            "features_dc": self.features_dc.data.detach().cpu(),
            "features_rest": self.features_rest.data.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if "features_dc" in state:
            self.features_dc = nn.Parameter(state["features_dc"].to(self.device))
            self.features_rest = nn.Parameter(state["features_rest"].to(self.device))

    def export(self, output_dir: str) -> list[str]:
        from src.exporter import export_diffuse_texture, export_sh_channels
        import os

        tex = self.get_material_texture()
        paths = []

        diffuse_path = os.path.join(output_dir, "diffuse.png")
        export_diffuse_texture(tex, diffuse_path, self.sh_order)
        paths.append(diffuse_path)

        sh_dir = os.path.join(output_dir, "sh_channels")
        paths.extend(export_sh_channels(tex, sh_dir, self.sh_order))

        return paths
