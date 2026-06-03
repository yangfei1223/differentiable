"""数据集模块 — GT 图像数据集，通过 image_path 精确关联图像与相机。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.camera import Camera, load_cameras


class GTDataset(Dataset):
    """Ground-Truth 图像数据集。

    从 cameras.json 加载相机列表，并通过每个条目的 ``image_path`` 字段
    精确关联对应的 GT 图像。

    Parameters
    ----------
    gt_dir : str
        GT 图像根目录（当 image_path 为相对路径时的基准目录）。
    camera_path : str
        cameras.json 文件路径。
    image_size : tuple[int, int] | None
        可选的统一缩放尺寸 (W, H)。若为 None 则使用原始尺寸。
    """

    def __init__(
        self,
        gt_dir: str,
        camera_path: str,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.gt_dir = Path(gt_dir)
        self.camera_path = Path(camera_path)
        self.image_size = image_size

        # 加载相机列表
        self.cameras: List[Camera] = load_cameras(self.camera_path)

        # 解析 cameras.json 获取 image_path 列表
        with open(self.camera_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cameras_raw = data["cameras"]
        self.image_paths: List[Path] = []
        for entry in cameras_raw:
            img_path = Path(entry["image_path"])
            if not img_path.is_absolute():
                # 相对于 cameras.json 所在目录解析
                img_path = self.camera_path.parent / img_path
            self.image_paths.append(img_path)

        # 校验数量一致性
        if len(self.image_paths) != len(self.cameras):
            raise ValueError(
                f"图像数量 ({len(self.image_paths)}) 与相机数量 "
                f"({len(self.cameras)}) 不匹配"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Camera]:
        """返回 (image_tensor[3,H,W], Camera)。"""
        img_path = self.image_paths[idx]
        cam = self.cameras[idx]

        # BGR → RGB
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 可选缩放
        if self.image_size is not None:
            w, h = self.image_size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # HWC → CHW，归一化到 [0, 1] float32
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return img, cam
