"""ShadingModel 基类 — 定义着色模型的接口协议。"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ShadingModel:
    """着色模型基类。"""

    def parameters(self) -> list[nn.Parameter]:
        raise NotImplementedError

    def init_textures(self, resolution: int) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    def get_material_texture(self) -> torch.Tensor:
        raise NotImplementedError

    def set_material_texture(self, texture: torch.Tensor) -> None:
        raise NotImplementedError

    def get_debug_info(self) -> dict:
        return {}

    def export(self, output_dir: str) -> list[str]:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
