"""配置系统 — 从 YAML 加载并校验参数。"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PBRConfig:
    env_map_res: list = field(default_factory=lambda: [256, 512])
    n_mip_levels: int = 5
    brdf_lut_size: int = 256
    env_lr_ratio: float = 1.0
    env_tv_weight: float = 0.0005
    env_l2_weight: float = 0.0001
    init_env_map: Optional[str] = None
    disable_normal_map: bool = False
    normal_map_path: Optional[str] = None  # 外部法线贴图路径（单 mesh 场景）


@dataclass
class NeuralLightmapConfig:
    feature_dim: int = 12              # 特征维度 C
    pe_level: int = 2                  # 视角 PE 阶数 L（→ 15D）
    encoding_mode: str = "reflect"     # "view"=PE(V) | "reflect"=PE(R)+NdotV
    mlp_hidden_dim: int = 32           # MLP 隐藏层宽度
    feature_lr: float = 0.1            # 特征纹理学习率（TTUR 大值）
    mlp_lr: float = 0.001              # MLP 学习率（TTUR 小值）
    feature_tv_weight: float = 0.00001 # 特征图 TV 正则
    feature_init_std: float = 0.1      # 初始化标准差


@dataclass
class DataConfig:
    mesh_path: str = "data/scene/lowpoly.obj"
    gt_dir: str = "data/gt"
    camera_path: str = "data/cameras.json"


@dataclass
class TextureConfig:
    sh_order: int = 2
    base_resolution: int = 512
    target_resolution: int = 4096
    init_dc_value: float = 0.5

    @property
    def num_sh_coeffs(self) -> int:
        return (self.sh_order + 1) ** 2

    @property
    def num_channels(self) -> int:
        return self.num_sh_coeffs * 3


@dataclass
class ResolutionStep:
    epoch: int = 0
    resolution: int = 512


@dataclass
class TrainingConfig:
    num_epochs: int = 2000
    lr: float = 0.01
    rest_lr_ratio: float = 0.05     # 高阶 SH 学习率 = lr * rest_lr_ratio
    lr_decay: float = 0.5
    lr_decay_epochs: List[int] = field(default_factory=lambda: [500, 1000, 1500])
    batch_size: int = 4
    resolution_schedule: List[ResolutionStep] = field(default_factory=lambda: [
        ResolutionStep(0, 512),
        ResolutionStep(300, 1024),
        ResolutionStep(700, 2048),
        ResolutionStep(1100, 4096),
    ])


@dataclass
class LossConfig:
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.2
    lambda_tv: float = 0.005


@dataclass
class SeamPaddingConfig:
    dilation_radius: int = 3
    apply_every_n_epochs: int = 50


@dataclass
class ExportConfig:
    output_dir: str = "output"
    format: str = "gltf"
    sh_truncate_order: int = -1


@dataclass
class VideoConfig:
    center: list[float] | None = None      # None = auto from mesh
    radius: float | None = None             # None = auto from mesh
    height: float | None = None             # None = auto from mesh
    fov_deg: float | None = None            # None = auto from mesh
    num_frames: int = 120
    resolution: int = 1024
    fps: int = 30


@dataclass
class Config:
    render_mode: str = "sh"  # "sh" | "pbr"
    pbr: PBRConfig = field(default_factory=PBRConfig)
    nlm: NeuralLightmapConfig = field(default_factory=NeuralLightmapConfig)
    data: DataConfig = field(default_factory=DataConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    seam_padding: SeamPaddingConfig = field(default_factory=SeamPaddingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


def _parse_resolution_schedule(raw: list) -> List[ResolutionStep]:
    return [ResolutionStep(**s) for s in raw]


def load_config(path: str | Path) -> Config:
    """从 YAML 文件加载配置。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()
    if "render_mode" in raw:
        cfg.render_mode = raw["render_mode"]
    if "pbr" in raw:
        cfg.pbr = PBRConfig(**raw["pbr"])
    if "nlm" in raw:
        cfg.nlm = NeuralLightmapConfig(**raw["nlm"])
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    if "texture" in raw:
        cfg.texture = TextureConfig(**raw["texture"])
    if "training" in raw:
        t = raw["training"]
        if "resolution_schedule" in t:
            t["resolution_schedule"] = _parse_resolution_schedule(t["resolution_schedule"])
        cfg.training = TrainingConfig(**t)
    if "loss" in raw:
        cfg.loss = LossConfig(**raw["loss"])
    if "seam_padding" in raw:
        cfg.seam_padding = SeamPaddingConfig(**raw["seam_padding"])
    if "export" in raw:
        cfg.export = ExportConfig(**raw["export"])
    if "video" in raw:
        cfg.video = VideoConfig(**raw["video"])

    return cfg
