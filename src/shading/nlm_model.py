"""Neural Lightmap shading model.

Per-submesh learnable feature map + shared TinyMLP decoder.
L_o(p, ω_o) = MLP( T(u,v), γ(ω_o) )
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn

from src.config import Config
from src.shading.base import ShadingModel
from src.shading.nlm.feature_map import init_feature_map
from src.shading.nlm.tiny_mlp import TinyMLP
from src.shading.nlm.positional_encode import positional_encode


class NeuralLightmapShadingModel(ShadingModel):
    """Neural Lightmap shading model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        nlm = config.nlm
        self.feature_dim = nlm.feature_dim
        self.pe_level = nlm.pe_level
        self.pe_dim = 3 * (1 + 2 * nlm.pe_level)  # L=2 -> 15
        self.mlp_in_dim = self.feature_dim + self.pe_dim  # 12 + 15 = 27

        self.feature_maps: dict[str, nn.Parameter] = {}
        self.mlp: TinyMLP = TinyMLP(
            in_dim=self.mlp_in_dim,
            hidden_dim=nlm.mlp_hidden_dim,
            out_dim=3,
        ).to(self.device)
        self.is_multi: bool = False
        self.resolution: int = config.texture.base_resolution

    # ------------------------------------------------------------------
    # Parameters & initialization
    # ------------------------------------------------------------------
    def parameters(self) -> list[nn.Parameter]:
        # TTUR: feature maps first (lr=1e-1), MLP params second (lr=1e-3)
        return list(self.feature_maps.values()) + list(self.mlp.parameters())

    def init_textures(self, resolution: int, submesh_names: list[str] | None = None) -> None:
        self.resolution = resolution
        nlm = self.config.nlm
        if submesh_names is not None:
            self.is_multi = True
            self.feature_maps = {
                name: nn.Parameter(
                    init_feature_map(resolution, nlm.feature_dim, nlm.feature_init_std).to(self.device)
                )
                for name in submesh_names
            }
        else:
            self.is_multi = False
            self.feature_maps = {
                "__default__": nn.Parameter(
                    init_feature_map(resolution, nlm.feature_dim, nlm.feature_init_std).to(self.device)
                )
            }

    # ------------------------------------------------------------------
    # Shading
    # ------------------------------------------------------------------
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
        """Single-mesh shade (delegates to shade_submesh with __default__)."""
        return self.shade_submesh(
            "__default__", rast_out, texc, world_pos, normals, view_dirs,
            camera, resolution, tangents, bitangents,
        )

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
        """Shade a submesh: sample feature -> PE(view) -> MLP -> scatter back."""
        import nvdiffrast.torch as dr

        H, W = resolution, resolution
        # 1. Sample feature texture via UV
        feature = dr.texture(
            self.feature_maps[name], texc, filter_mode="linear", boundary_mode="clamp"
        )  # [1, H, W, C]

        # 2. Mask valid pixels
        mask = (rast_out[..., 3] > 0)  # [1, H, W] bool
        if mask.sum() == 0:
            rgb = torch.zeros(1, H, W, 3, device=self.device)
            return rgb, mask.float()

        # 3. Index valid pixels only (save ~80% FLOPs on background)
        feat_valid = feature[mask]                     # [N, C]
        view_valid = view_dirs[mask]                   # [N, 3]

        # 4. Positional encode view direction
        view_pe = positional_encode(view_valid, self.pe_level)  # [N, pe_dim]

        # 5. Concat & decode
        x = torch.cat([feat_valid, view_pe], dim=-1)   # [N, C+pe_dim]
        rgb_valid = self.mlp(x)                        # [N, 3], Softplus >= 0

        # 6. Scatter back to full image
        rgb = torch.zeros(1, H, W, 3, device=self.device)
        rgb[mask] = rgb_valid

        # Debug info (feature first 3 channels as visualization proxy)
        self._last_debug = {"feature": feature.detach()}

        return rgb, mask.float()

    # ------------------------------------------------------------------
    # Material texture accessors (for resize / seam padding)
    # ------------------------------------------------------------------
    def get_material_texture(self) -> dict[str, torch.Tensor]:
        return {k: v.data.detach().cpu() for k, v in self.feature_maps.items()}

    def set_material_texture(self, texture) -> None:
        if isinstance(texture, dict):
            self.is_multi = True
            self.feature_maps = {
                k: nn.Parameter(v.to(self.device).contiguous()) for k, v in texture.items()
            }
        else:
            # Single tensor -- wrap in default key
            self.is_multi = False
            self.feature_maps = {
                "__default__": nn.Parameter(texture.to(self.device).contiguous())
            }

    def get_debug_info(self) -> dict:
        return getattr(self, "_last_debug", {})

    # ------------------------------------------------------------------
    # Multi-mesh training hooks
    # ------------------------------------------------------------------
    def regularization_loss(self) -> torch.Tensor:
        """NLM has no global regularization (feature TV is per-submesh)."""
        return torch.tensor(0.0, device=self.device)

    def get_submesh_texture(self, name: str) -> torch.Tensor:
        return self.feature_maps[name]

    def post_backward_hook(self) -> None:
        """NLM requires no post-backward cleanup."""
        pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "render_mode": "nlm",
            "is_multi": self.is_multi,
            "feature_maps": {k: v.data.detach().cpu() for k, v in self.feature_maps.items()},
            "mlp_state": self.mlp.state_dict(),
            "resolution": self.resolution,
        }

    def load_state_dict(self, state: dict) -> None:
        self.is_multi = state.get("is_multi", True)
        self.resolution = state.get("resolution", self.config.texture.base_resolution)
        self.feature_maps = {
            k: nn.Parameter(v.to(self.device)) for k, v in state["feature_maps"].items()
        }
        if "mlp_state" in state:
            self.mlp.load_state_dict(state["mlp_state"])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, output_dir: str) -> list[str]:
        import numpy as np
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        paths = []

        # Feature map PNG (first 3 channels) + PT (full)
        for name, fm in self.feature_maps.items():
            sub_dir = output_dir if not self.is_multi else os.path.join(output_dir, name)
            if self.is_multi:
                os.makedirs(sub_dir, exist_ok=True)

            # PNG: first 3 channels normalized to [0,255]
            vis = fm[0, ..., :3].clamp(-1, 1)  # feature may be negative
            vis = ((vis + 1) * 0.5 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            png_path = os.path.join(sub_dir, f"feature_map_{name}.png")
            Image.fromarray(vis, "RGB").save(png_path)
            paths.append(png_path)

            # PT: full float32 tensor
            pt_path = os.path.join(sub_dir, f"feature_map_{name}.pt")
            torch.save(fm.data.detach().cpu(), pt_path)
            paths.append(pt_path)

        # MLP weights
        mlp_path = os.path.join(output_dir, "mlp_weights.pt")
        torch.save(self.mlp.state_dict(), mlp_path)
        paths.append(mlp_path)

        return paths
