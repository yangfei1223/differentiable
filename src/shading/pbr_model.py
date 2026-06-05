"""PBR 着色模型。"""
from __future__ import annotations

from src.shading.base import ShadingModel


class PBRShadingModel(ShadingModel):
    def __init__(self, config):
        self.config = config
