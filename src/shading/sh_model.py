"""SH 着色模型包装。"""
from __future__ import annotations

from src.shading.base import ShadingModel


class SHShadingModel(ShadingModel):
    def __init__(self, config):
        self.config = config
