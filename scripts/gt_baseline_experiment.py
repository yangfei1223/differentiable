"""GT 材质基线实验 — GT 材质 + EXR 环境光初始化，正常 PBR 训练摸上限。

对比基线：之前 random init 的头盔 PBR 训练达到 21.97 dB。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    Config, DataConfig, PBRConfig, TextureConfig,
    TrainingConfig, LossConfig, SeamPaddingConfig, ResolutionStep,
)
from src.mesh import load_mesh
from src.shading.pbr_model import PBRShadingModel
from src.shading.pbr.material import decode_material
from src.shading.pbr.env_map import EnvironmentMap
from src.dataset import GTDataset
from src.renderer import DifferentiableRenderer
from src.trainer import Trainer


# ==========================================================================
# GT 材质加载
# ==========================================================================
def load_gt_material_texture(tex_dir: str, resolution: int = 2048) -> torch.Tensor:
    """从 GT 贴图构建 8ch sigmoid 编码的材质参数 [1, H, W, 8]。"""
    from PIL import Image

    # Base color: sRGB → linear → sigmoid inverse
    albedo = np.array(Image.open(os.path.join(tex_dir, "Default_albedo.jpg")).convert("RGB")).astype(np.float32) / 255.0
    albedo_linear = np.power(albedo, 2.2)
    if albedo_linear.shape[0] != resolution:
        albedo_linear = cv2.resize(albedo_linear, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

    # Metallic & Roughness: glTF ORM — G=roughness, B=metallic
    mr = np.array(Image.open(os.path.join(tex_dir, "Default_metalRoughness.jpg")).convert("RGB")).astype(np.float32) / 255.0
    if mr.shape[0] != resolution:
        mr = cv2.resize(mr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    roughness = mr[..., 1:2]
    metallic = mr[..., 2:3]

    # Normal: [0,1] → [-1,1] → normalize
    normal = np.array(Image.open(os.path.join(tex_dir, "Default_normal.jpg")).convert("RGB")).astype(np.float32) / 255.0
    if normal.shape[0] != resolution:
        normal = cv2.resize(normal, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    normal = normal * 2.0 - 1.0
    normal = normal / np.sqrt(np.sum(normal ** 2, axis=-1, keepdims=True)).clip(1e-8)

    # Sigmoid inverse: log(x / (1-x))
    eps = 1e-6
    base_raw = np.log(albedo_linear.clip(eps, 1 - eps) / (1 - albedo_linear.clip(eps, 1 - eps)))
    rough_raw = np.log(roughness.clip(eps, 1 - eps) / (1 - roughness.clip(eps, 1 - eps)))
    metal_raw = np.log(metallic.clip(eps, 1 - eps) / (1 - metallic.clip(eps, 1 - eps)))

    raw = np.concatenate([base_raw, rough_raw, metal_raw, normal], axis=-1).astype(np.float32)
    raw_t = torch.from_numpy(raw).unsqueeze(0)

    # Verify
    bc, r, m, n = decode_material(raw_t)
    print(f"[GT Material] base_color: [{bc.min():.3f}, {bc.max():.3f}], "
          f"roughness: [{r.min():.3f}, {r.max():.3f}], "
          f"metallic: [{m.min():.3f}, {m.max():.3f}]")
    return raw_t


def load_gt_env_map(exr_path: str, target_h: int = 256, target_w: int = 512) -> torch.Tensor:
    """从 EXR 加载 HDR 环境光，softplus 编码 [1, H, W, 3]。"""
    img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"[GT EnvMap] EXR: shape={img.shape}, min={img.min():.3f}, max={img.max():.1f}")

    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    img = img.clip(min=1e-6, max=10.0)  # clamp 避免 softplus 溢出
    img_t = torch.from_numpy(img).unsqueeze(0).float()
    raw = torch.log(torch.exp(img_t) - 1.0)

    decoded = F.softplus(raw)
    print(f"[GT EnvMap] Decoded: [{decoded.min():.3f}, {decoded.max():.3f}]")
    return raw


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="GT Material Init PBR Training")
    parser.add_argument("--mesh_path", default="data/helmet_260604/scene/lowpoly.glb")
    parser.add_argument("--gt_dir", default="data/helmet_260604/gt")
    parser.add_argument("--camera_path", default="data/helmet_260604/cameras.json")
    parser.add_argument("--tex_dir", default="data/helmet_260604/scene/gt_textures")
    parser.add_argument("--env_path", default="asset/interior.exr")
    parser.add_argument("--tex_res", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output_dir", default="output/helmet_gt_init")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 GT 材质 + 环境光
    gt_mat = load_gt_material_texture(args.tex_dir, resolution=args.tex_res)
    gt_env = load_gt_env_map(args.env_path)

    # 2. 构建 Config（和 train_pbr.yaml 一致的训练策略）
    cfg = Config(
        render_mode="pbr",
        data=DataConfig(mesh_path=args.mesh_path, gt_dir=args.gt_dir, camera_path=args.camera_path),
        pbr=PBRConfig(env_map_res=[256, 512], brdf_lut_size=256,
                       env_tv_weight=0.0005, env_l2_weight=0.0001),
        texture=TextureConfig(base_resolution=args.tex_res),
        training=TrainingConfig(
            num_epochs=args.num_epochs, lr=args.lr, batch_size=4,
            lr_decay=0.5,
            lr_decay_epochs=[500, 1000, 1500],
            resolution_schedule=[
                ResolutionStep(0, 512),
                ResolutionStep(300, 1024),
                ResolutionStep(700, 2048),
            ],
        ),
        loss=LossConfig(lambda_l1=1.0, lambda_ssim=0.2, lambda_tv=0.005),
        seam_padding=SeamPaddingConfig(dilation_radius=3, apply_every_n_epochs=50),
    )

    # 3. 创建模型，GT 初始化
    model = PBRShadingModel(cfg)
    model.init_textures(args.tex_res)
    model.mat_texture = nn.Parameter(gt_mat.to(device).contiguous())
    model.env_map = EnvironmentMap(height=256, width=512).to(device)
    model.env_map.raw = nn.Parameter(gt_env.to(device).contiguous())

    # 4. Trainer
    trainer = Trainer(cfg, shading_model=model)

    print(f"\n--- Training {args.num_epochs} epochs, output → {args.output_dir} ---")
    trainer.train(
        output_dir=args.output_dir,
        checkpoint_every=200,
    )

    # 5. 导出最终材质
    mat_dir = os.path.join(args.output_dir, "final_materials")
    os.makedirs(mat_dir, exist_ok=True)
    trainer.model.export(mat_dir)
    print(f"Final materials exported to {mat_dir}/")


if __name__ == "__main__":
    main()
