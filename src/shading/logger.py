"""着色模型调试日志 — 基类与工厂函数。"""
from __future__ import annotations

import os
from typing import Optional

import torch

from src.config import Config


class ShadingLogger:
    """着色模型调试日志基类。"""

    def __init__(self, config: Config):
        self.config = config

    def save_checkpoint(
        self,
        model,
        output_dir: str,
        epoch: int,
        loss: float,
        resolution: int,
    ) -> str:
        """保存 checkpoint，返回文件路径。"""
        raise NotImplementedError

    def export_debug(
        self,
        model,
        renderer,
        dataset,
        output_dir: str,
        epoch: int,
        history: dict,
        device: str,
        current_resolution: int,
    ) -> None:
        """导出调试可视化（compare 图 + 纹理 + 视频）。"""
        raise NotImplementedError


def create_logger(render_mode: str, config: Config) -> ShadingLogger:
    """根据 render_mode 创建对应的 logger。"""
    if render_mode == "sh":
        from src.shading.sh_logger import SHLogger
        return SHLogger(config)
    elif render_mode == "pbr":
        from src.shading.pbr_logger import PBRLogger
        return PBRLogger(config)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode!r}")
