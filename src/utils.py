"""调试可视化工具 — 基于 OpenCV 的中间数据可视化。

支持 numpy ndarray 和 torch Tensor，自动处理：
- CHW ↔ HWC 转换
- linear → sRGB gamma 校正（可选）
- float [0,1] → uint8 范围映射
- 多图拼接（横排/竖排/网格）
- 像素值统计叠加显示
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy_hwc_uint8(
    img: ArrayLike,
    gamma: bool = False,
) -> np.ndarray:
    """将各种格式的图像数据转为 [H, W, C] uint8 numpy 数组。

    Args:
        img: 输入数据，支持：
            - numpy: [H,W], [H,W,1], [H,W,3], [1,H,W], [3,H,W], [1,H,W,3]
            - torch: 同上形状的 Tensor
        gamma: 是否应用 linear → sRGB gamma (pow 1/2.2)
    """
    if HAS_TORCH and isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.asarray(img, dtype=np.float32)

    # 去掉多余的前导维度 [1, ...]
    while img.ndim > 3 and img.shape[0] == 1:
        img = img[0]

    # CHW → HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
        img = img.transpose(1, 2, 0)

    # [H, W, 1] → [H, W]
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # 归一化到 [0, 1]
    if img.max() > 1.0 or img.min() < 0.0:
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

    img = np.ascontiguousarray(np.clip(img, 0.0, 1.0))

    if gamma:
        img = np.ascontiguousarray(np.power(img, 1.0 / 2.2))

    # 灰度 → 3 通道（方便叠加文字/拼接）
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return np.ascontiguousarray((img * 255).astype(np.uint8))


def vis(
    img: ArrayLike,
    title: Optional[str] = None,
    gamma: bool = False,
    stats: bool = True,
    block: bool = True,
    win_name: Optional[str] = None,
) -> None:
    """显示单张图像（cv2.imshow）。

    Args:
        img: 图像数据（numpy / torch）。
        title: 图像上方叠加的标题文字。
        gamma: 是否应用 gamma 校正。
        stats: 是否叠加像素值统计（min/max/mean）。
        block: 是否阻塞等待按键。
        win_name: 窗口名（默认用 title 或 "vis"）。
    """
    frame = _to_numpy_hwc_uint8(img, gamma=gamma)

    # 统计叠加
    if stats:
        raw = img.detach().cpu().numpy() if (HAS_TORCH and isinstance(img, torch.Tensor)) else np.asarray(img, dtype=np.float32)
        info = f"shape={list(raw.shape)} min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}"
        cv2.putText(frame, info, (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if title:
        cv2.putText(frame, title, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    name = win_name or title or "vis"
    cv2.imshow(name, frame)
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def vis_grid(
    imgs: List[ArrayLike],
    titles: Optional[List[str]] = None,
    cols: int = 0,
    gamma: bool = False,
    stats: bool = True,
    block: bool = True,
    win_name: str = "grid",
) -> None:
    """多图网格显示。

    Args:
        imgs: 图像列表。
        titles: 对应标题列表。
        cols: 列数，0 = 自动（尽量接近正方形）。
        gamma: 是否 gamma 校正。
        stats: 是否叠加统计。
        block: 是否阻塞。
        win_name: 窗口名。
    """
    n = len(imgs)
    if n == 0:
        return
    if cols <= 0:
        cols = max(1, int(np.ceil(np.sqrt(n))))

    frames = []
    for i, img in enumerate(imgs):
        frame = _to_numpy_hwc_uint8(img, gamma=gamma)

        if titles and i < len(titles):
            cv2.putText(frame, titles[i], (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if stats:
            raw = img.detach().cpu().numpy() if (HAS_TORCH and isinstance(img, torch.Tensor)) else np.asarray(img, dtype=np.float32)
            info = f"min={raw.min():.3f} max={raw.max():.3f}"
            cv2.putText(frame, info, (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        frames.append(frame)

    # 统一高度为最大高度
    max_h = max(f.shape[0] for f in frames)
    padded = []
    for f in frames:
        if f.shape[0] < max_h:
            pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f = np.concatenate([f, pad], axis=0)
        padded.append(f)

    # 拼接网格
    rows_data = []
    for r in range(0, n, cols):
        row_imgs = padded[r:r + cols]
        # 不足 cols 的补黑
        while len(row_imgs) < cols:
            h, w = row_imgs[0].shape[:2]
            row_imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows_data.append(np.concatenate(row_imgs, axis=1))

    grid = np.concatenate(rows_data, axis=0)

    cv2.imshow(win_name, grid)
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def vis_pair(
    a: ArrayLike,
    b: ArrayLike,
    label_a: str = "A",
    label_b: str = "B",
    gamma: bool = False,
    block: bool = True,
    win_name: str = "pair",
) -> None:
    """并排显示两张图像。

    Args:
        a, b: 两张图像。
        label_a, label_b: 标签。
        gamma: 是否 gamma 校正。
        block: 是否阻塞。
        win_name: 窗口名。
    """
    fa = _to_numpy_hwc_uint8(a, gamma=gamma)
    fb = _to_numpy_hwc_uint8(b, gamma=gamma)

    # 统一高度
    h = max(fa.shape[0], fb.shape[0])
    for f in [fa, fb]:
        if f.shape[0] < h:
            pad = np.zeros((h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f_new = np.concatenate([f, pad], axis=0)
            if f is fa:
                fa = f_new
            else:
                fb = f_new

    cv2.putText(fa, label_a, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(fb, label_b, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    canvas = np.concatenate([fa, fb], axis=1)
    cv2.imshow(win_name, canvas)
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save(
    img: ArrayLike,
    path: str,
    gamma: bool = False,
) -> str:
    """保存图像到文件。

    Args:
        img: 图像数据。
        path: 输出路径。
        gamma: 是否 gamma 校正。

    Returns:
        保存路径。
    """
    frame = _to_numpy_hwc_uint8(img, gamma=gamma)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, frame_bgr)
    return path
