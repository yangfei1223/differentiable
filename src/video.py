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
    radius: Optional[float] = None,
    height: Optional[float] = None,
    fov_deg: Optional[float] = None,
    num_frames: int = 120,
    resolution: int = 1024,
    fps: int = 30,
    device: str = "cuda",
    fill_ratio: float = 0.6,
    subtract_texture: Optional[torch.Tensor] = None,
) -> str:
    """用训练好的 SH 纹理渲染一段轨道视频。

    相机参数（center, radius, height, fov_deg）未指定时
    从 mesh bounding box 自动计算，确保物体在画面中占合理比例。

    Args:
        sh_texture: SH 系数纹理 [1, H, W, C]。
        mesh: 网格数据。
        output_path: 输出 mp4 路径。
        center: 注视中心 [x, y, z]。None 则自动计算。
        radius: 水平环绕半径。None 则自动计算。
        height: 相机高度。None 则自动计算。
        fov_deg: 垂直视场角。None 则自动计算。
        num_frames: 帧数。
        resolution: 渲染分辨率。
        fps: 帧率。
        device: 设备。
        fill_ratio: 物体在画面中的占比（0~1），用于自动计算 FOV。
        subtract_texture: 若提供，渲染结果减去该纹理的渲染（净值）。
        sh_texture: SH 系数纹理 [1, H, W, C]。
        mesh: 网格数据。
        output_path: 输出 mp4 路径。
        center: 注视中心 [x, y, z]。None 则自动计算。
        radius: 水平环绕半径。None 则自动计算。
        height: 相机高度。None 则自动计算。
        fov_deg: 垂直视场角。None 则自动计算。
        num_frames: 帧数。
        resolution: 渲染分辨率。
        fps: 帧率。
        device: 设备。
        fill_ratio: 物体在画面中的占比（0~1），用于自动计算 FOV。

    Returns:
        输出视频文件的绝对路径。
    """
    # ---- 1. 从 mesh bounding box 自动计算相机参数 ----
    verts_np = mesh.vertices

    # bbox
    v_min = verts_np.min(axis=0)
    v_max = verts_np.max(axis=0)
    bbox_size = v_max - v_min
    bbox_max_dim = float(bbox_size.max())

    # center
    if center is None:
        center_np = (v_min + v_max) / 2.0
    else:
        center_np = np.array(center, dtype=np.float64)

    # 包围球半径
    dists = np.linalg.norm(verts_np - center_np, axis=1)
    bsphere_radius = float(dists.max())

    # 相机距离 = 包围球半径 / sin(fov/2) / fill_ratio
    # 先确定 radius 和 height，再算 fov
    if radius is None:
        radius = bsphere_radius * 2.5  # 相机离中心约 2.5 倍包围球半径

    if height is None:
        height = center_np[1] + bsphere_radius * 1.2  # 比模型顶部略高

    if fov_deg is None:
        # 从相机到 bbox 中心的最大距离，算 FOV 使物体占 fill_ratio
        cam_y = height
        cam_dist = math.sqrt(radius ** 2 + (cam_y - center_np[1]) ** 2)
        # 物体半角 = atan(bsphere_radius / cam_dist)
        half_angle = math.atan(bsphere_radius / cam_dist)
        # fill_ratio = 2 * half_angle / fov → fov = 2 * half_angle / fill_ratio
        fov_deg = math.degrees(2.0 * half_angle / fill_ratio)
        # 限制合理范围
        fov_deg = max(20.0, min(fov_deg, 70.0))

    print(f"  [Video] Auto camera: radius={radius:.2f}, height={height:.2f}, fov={fov_deg:.1f}°, "
          f"bbox={bbox_max_dim:.2f}, bsphere={bsphere_radius:.2f}")

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
    verts, faces, uvs, uv_idx, normals, normal_idx = mesh.to_torch()
    renderer = DifferentiableRenderer(
        vertices=verts,
        faces=faces,
        uvs=uvs,
        uv_idx=uv_idx,
        normals=normals,
        normal_idx=normal_idx,
        resolution=resolution,
        device=device,
    )

    # sh_texture is the full concatenated texture [1, H, W, n*3]
    # Split into DC + Rest for renderer
    n_dc = 3
    features_dc = sh_texture[..., :n_dc]
    features_rest = sh_texture[..., n_dc:]

    dc_param = nn.Parameter(features_dc.to(device))
    rest_param = nn.Parameter(features_rest.to(device))

    # subtract_texture: 渲染减去该纹理的结果（高频净值）
    sub_dc_param = None
    sub_rest_param = None
    if subtract_texture is not None:
        sub_dc = subtract_texture[..., :n_dc]
        sub_rest = subtract_texture[..., n_dc:]
        sub_dc_param = nn.Parameter(sub_dc.to(device))
        sub_rest_param = nn.Parameter(sub_rest.to(device))

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
            rgb, mask, _ = renderer.render(dc_param, rest_param, cam)  # [1, H, W, 3]

            # 减去 subtract_texture 的渲染结果
            if sub_dc_param is not None:
                rgb_sub, _, _ = renderer.render(sub_dc_param, sub_rest_param, cam)
                rgb = (rgb - rgb_sub).clamp(0.0, 1.0)

        # [1, H, W, 3] → [H, W, 3] numpy uint8
        # nvdiffrast 输出为 OpenGL 坐标 (原点左下)，需垂直翻转为图像坐标 (原点左上)
        # 渲染输出是线性空间, 需 gamma 校正到 sRGB
        frame = rgb[0].detach().cpu().flip(0).clamp(0.0, 1.0).pow(1.0 / 2.2).numpy()
        frame = (frame * 255).astype(np.uint8)
        # RGB → BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 黑色背景 (mask 为 0 的区域)
        mask_np = mask[0].detach().cpu().flip(0).numpy()  # [H, W]，同样翻转
        bg = mask_np < 0.5  # [H, W]
        frame[bg] = 0  # broadcast to [H, W, 3]

        writer.write(frame)

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  [Video] Frame {i+1}/{num_frames}")

    writer.release()
    print(f"  [Video] Saved: {output_path}")
    return output_path
