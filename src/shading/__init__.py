"""着色模型可插拔层 — 工厂函数。"""
from __future__ import annotations

from src.config import Config


def create_shading_model(render_mode: str, config: Config):
    """根据 render_mode 创建对应的着色模型。"""
    if render_mode == "sh":
        from src.shading.sh_model import SHShadingModel
        return SHShadingModel(config)
    elif render_mode == "pbr":
        from src.shading.pbr_model import PBRShadingModel
        return PBRShadingModel(config)
    elif render_mode == "nlm":
        from src.shading.nlm_model import NeuralLightmapShadingModel
        return NeuralLightmapShadingModel(config)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode!r}")
