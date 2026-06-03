"""数据集模块测试 — GTDataset 加载与索引一致性。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.camera import Camera


# ---------------------------------------------------------------------------
# 测试数据生成辅助函数
# ---------------------------------------------------------------------------
def _make_test_data(tmp_path: Path, num_views: int = 3, size: int = 64):
    """在 tmp_path 下创建 gt/ 图像目录和 cameras.json。

    Parameters
    ----------
    tmp_path : Path
        临时目录根。
    num_views : int
        视角数量。
    size : int
        生成的正方形图像边长。

    Returns
    -------
    gt_dir : Path
        gt 图像目录路径。
    camera_path : Path
        cameras.json 路径。
    """
    import cv2

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    cameras = []
    for i in range(num_views):
        img_name = f"view_{i:04d}.png"
        # 生成随机 BGR 图像 (uint8)
        img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(gt_dir / img_name), img)

        cameras.append({
            "image_path": f"gt/{img_name}",
            "position": [float(i), 0.0, 5.0],
            "look_at": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 60.0,
            "image_size": [size, size],
        })

    cameras_data = {
        "blender_coordinate": False,
        "cameras": cameras,
    }
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(json.dumps(cameras_data, indent=2), encoding="utf-8")

    return gt_dir, camera_path


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
class TestGTDataset:
    """GTDataset 核心功能测试。"""

    def test_dataset_length(self, tmp_path):
        """GTDataset.__len__ 应返回 cameras.json 中的相机数量。"""
        from src.dataset import GTDataset

        gt_dir, camera_path = _make_test_data(tmp_path, num_views=5, size=32)
        dataset = GTDataset(gt_dir=str(gt_dir), camera_path=str(camera_path))
        assert len(dataset) == 5

    def test_dataset_item_shapes(self, tmp_path):
        """每个 item 应返回 (tensor[3,H,W], Camera)，图像值在 [0,1]。"""
        from src.dataset import GTDataset

        size = 48
        gt_dir, camera_path = _make_test_data(tmp_path, num_views=3, size=size)
        dataset = GTDataset(gt_dir=str(gt_dir), camera_path=str(camera_path))

        img_tensor, cam = dataset[0]

        # 类型与形状
        assert img_tensor.shape == (3, size, size)
        assert isinstance(cam, Camera)

        # 值域 [0, 1]
        assert img_tensor.min() >= 0.0
        assert img_tensor.max() <= 1.0
        assert img_tensor.dtype.name == "float32"

    def test_dataset_iterate_all(self, tmp_path):
        """遍历所有 item 不应抛出异常。"""
        from src.dataset import GTDataset

        gt_dir, camera_path = _make_test_data(tmp_path, num_views=4, size=16)
        dataset = GTDataset(gt_dir=str(gt_dir), camera_path=str(camera_path))

        results = []
        for i in range(len(dataset)):
            item = dataset[i]
            results.append(item)

        assert len(results) == 4
        # 每个 item 都应是 (tensor, Camera) 二元组
        for img_tensor, cam in results:
            assert img_tensor.shape[0] == 3  # 通道
            assert isinstance(cam, Camera)
