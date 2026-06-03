"""视频导出 — 高机位环绕轨道渲染。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.camera import Camera
from src.mesh import MeshData
from src.renderer import DifferentiableRenderer
from src.sh import eval_sh_basis


def orbit_cameras(
    center: np.ndarray = np.array([0.0, 2.0, 0.0]),
    radius: float = 14.0,
    height: float = 10.0,
    num_frames: int = 120,
    fov_deg: float = 45.0,
    resolution: int = 1024,
) -> list[Camera]:
    """生成高机位环绕轨道相机列表。

    相机在 OpenGL 坐标系 (Y-up) 下生成，从高处俯瞰模型中心，
    水平方向环绕 360°。

    Args:
        center: 注视目标（世界坐标）。
        radius: 水平环绕半径。
        height: 相机高度 (Y 坐标)。
        num_frames: 帧数（一圈）。
        fov_deg: 垂直视场角。
        resolution: 渲染分辨率 (正方形)。

    Returns:
        Camera 对象列表，长度 = num_frames。
    """
    cameras = []
    for i in range(num_frames):
        theta = 2.0 * math.pi * i / num_frames
        # OpenGL: Y-up, 相机朝 -Z
        # 在 XZ 平面环绕，Y 方向抬高
        x = center[0] + radius * math.sin(theta)
        y = height
        z = center[2] + radius * math.cos(theta)

        pos = np.array([x, y, z], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        cameras.append(Camera(
            position=pos,
            look_at=center.copy(),
            up=up,
            fov_deg=fov_deg,
            image_width=resolution,
            image_height=resolution,
        ))
    return cameras


def render_video(
    sh_texture: torch.Tensor,
    mesh: MeshData,
    output_path: str,
    center: Optional[list[float]] = None,
    radius: float = 14.0,
    height: float = 10.0,
    num_frames: int = 120,
    fov_deg: float = 45.0,
    resolution: int = 1024,
    fps: int = 30,
    device: str = "cuda",
) -> str:
    """用训练好的 SH 纹理渲染一段轨道视频。

    Args:
        sh_texture: SH 系数纹理 [1, H, W, C]。
        mesh: 网格数据。
        output_path: 输出 mp4 路径。
        center: 注视中心 [x, y, z]。None 则自动从网格计算。
        radius: 环绕半径。
        height: 相机高度。
        num_frames: 帧数。
        fov_deg: 视场角。
        resolution: 渲染分辨率。
        fps: 帧率。
        device: 设备。

    Returns:
        输出视频文件的绝对路径。
    """
    # ---- 1. 自动计算 center ----
    if center is None:
        verts_np = mesh.vertices
        center_np = verts_np.mean(axis=0)
        # 偏向模型中部高度
        center_np[1] = (verts_np[:, 1].min() + verts_np[:, 1].max()) / 2.0
    else:
        center_np = np.array(center, dtype=np.float64)

    # ---- 2. 生成轨道相机 ----
    cameras = orbit_cameras(
        center=center_np,
        radius=radius,
        height=height,
        num_frames=num_frames,
        fov_deg=fov_deg,
        resolution=resolution,
    )

    # ---- 3. 创建渲染器 ----
    verts, faces, uvs, uv_idx = mesh.to_torch()
    renderer = DifferentiableRenderer(
        vertices=verts,
        faces=faces,
        uvs=uvs,
        uv_idx=uv_idx,
        resolution=resolution,
        device=device,
    )

    sh_param = nn.Parameter(sh_texture.data.to(device))

    # ---- 4. 创建视频写入器 ----
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (resolution, resolution))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    # ---- 5. 逐帧渲染 ----
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, mask = renderer.render(sh_param, cam)  # [1, H, W, 3]

        # [1, H, W, 3] → [H, W, 3] numpy uint8
        frame = rgb[0].detach().cpu().clamp(0.0, 1.0).numpy()
        frame = (frame * 255).astype(np.uint8)
        # RGB → BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 黑色背景 (mask 为 0 的区域)
        mask_np = mask[0].detach().cpu().numpy()  # [H, W]
        bg = mask_np < 0.5  # [H, W]
        frame[bg] = 0  # broadcast to [H, W, 3]

        writer.write(frame)

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  [Video] Frame {i+1}/{num_frames}")

    writer.release()
    print(f"  [Video] Saved: {output_path}")
    return output_path
