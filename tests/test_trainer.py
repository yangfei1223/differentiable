"""训练器集成测试（需要 CUDA + nvdiffrast）。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.config import (
    Config,
    DataConfig,
    TextureConfig,
    TrainingConfig,
    LossConfig,
    SeamPaddingConfig,
    ResolutionStep,
)

# ---------------------------------------------------------------------------
# CUDA 跳过标记
# ---------------------------------------------------------------------------
cuda_skip = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA",
)


# ---------------------------------------------------------------------------
# 测试数据辅助
# ---------------------------------------------------------------------------
def _make_training_data(tmp_path: Path, num_views: int = 4, img_size: int = 32):
    """在 tmp_path 下创建训练所需的 mesh、GT 图像和相机文件。

    Parameters
    ----------
    tmp_path : Path
        临时目录。
    num_views : int
        视角数量。
    img_size : int
        正方形图像边长。

    Returns
    -------
    mesh_path : Path
    gt_dir : Path
    camera_path : Path
    """
    # ---- 写平面 OBJ ----
    mesh_path = tmp_path / "mesh.obj"
    obj_content = (
        "# Simple quad plane with UVs\n"
        "v -1.0 -1.0 0.0\n"
        "v  1.0 -1.0 0.0\n"
        "v  1.0  1.0 0.0\n"
        "v -1.0  1.0 0.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 1.0 1.0\n"
        "vt 0.0 1.0\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n"
    )
    mesh_path.write_text(obj_content, encoding="utf-8")

    # ---- 创建 gt/ 目录和随机 PNG 图像 ----
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    cameras = []
    for i in range(num_views):
        img_name = f"view_{i:04d}.png"
        img = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        cv2.imwrite(str(gt_dir / img_name), img)

        # 相机从不同角度对准原点
        angle = 2.0 * np.pi * i / num_views
        cam_x = 3.0 * np.sin(angle)
        cam_z = 3.0 * np.cos(angle)

        cameras.append({
            "image_path": f"gt/{img_name}",
            "position": [float(cam_x), 0.0, float(cam_z)],
            "look_at": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 60.0,
            "image_size": [img_size, img_size],
        })

    cameras_data = {
        "blender_coordinate": False,
        "cameras": cameras,
    }
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(json.dumps(cameras_data, indent=2), encoding="utf-8")

    return mesh_path, gt_dir, camera_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cuda_skip
def test_trainer_runs(tmp_path):
    """Trainer.train() 应能完整运行，且 SH 纹理被更新。"""
    mesh_path, gt_dir, camera_path = _make_training_data(
        tmp_path, num_views=4, img_size=32,
    )

    cfg = Config(
        data=DataConfig(
            mesh_path=str(mesh_path),
            gt_dir=str(gt_dir),
            camera_path=str(camera_path),
        ),
        texture=TextureConfig(
            sh_order=2,
            base_resolution=16,
            target_resolution=16,
            init_dc_value=0.5,
        ),
        training=TrainingConfig(
            num_epochs=3,
            lr=0.01,
            lr_decay=0.5,
            lr_decay_epochs=[2],
            batch_size=2,
            resolution_schedule=[ResolutionStep(0, 16)],
        ),
        loss=LossConfig(
            lambda_l1=1.0,
            lambda_ssim=0.0,
            lambda_tv=0.0,
        ),
        seam_padding=SeamPaddingConfig(
            dilation_radius=1,
            apply_every_n_epochs=0,  # 禁用 seam padding
        ),
    )

    from src.trainer import Trainer

    trainer = Trainer(cfg)

    # 记录初始纹理
    initial_texture = trainer.get_sh_texture().clone()

    # 运行训练
    trainer.train()

    # 验证 SH 纹理已被更新（不是完全相同）
    final_texture = trainer.get_sh_texture()

    assert final_texture.shape == initial_texture.shape, "纹理形状不应改变"
    # 纹理应该发生了变化（至少某些像素不同）
    diff = (final_texture - initial_texture).abs().sum().item()
    assert diff > 0, "SH 纹理应在训练后发生变化"
