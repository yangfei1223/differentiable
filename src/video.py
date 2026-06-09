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
    mesh: MeshData,
    output_path: str,
    sh_texture: Optional[torch.Tensor] = None,
    shading_model=None,
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
    """用训练好的纹理/着色模型渲染一段轨道视频。

    支持两种渲染路径:
    - **shading_model** (新): 传入 ShadingModel 对象，使用
      ``renderer.rasterize_and_interpolate`` + ``shading_model.shade``。
    - **sh_texture** (旧): 传入 SH 系数纹理张量，使用传统
      ``renderer.render(dc_param, rest_param, cam)`` 路径。

    二者只需提供其一；若同时提供，优先使用 shading_model。

    相机参数（center, radius, height, fov_deg）未指定时
    从 mesh bounding box 自动计算，确保物体在画面中占合理比例。

    Args:
        mesh: 网格数据。
        output_path: 输出 mp4 路径。
        sh_texture: SH 系数纹理 [1, H, W, C]（旧路径）。
        shading_model: ShadingModel 实例（新路径）。
        center: 注视中心 [x, y, z]。None 则自动计算。
        radius: 水平环绕半径。None 则自动计算。
        height: 相机高度。None 则自动计算。
        fov_deg: 垂直视场角。None 则自动计算。
        num_frames: 帧数。
        resolution: 渲染分辨率。
        fps: 帧率。
        device: 设备。
        fill_ratio: 物体在画面中的占比（0~1），用于自动计算 FOV。
        subtract_texture: 若提供，渲染结果减去该纹理的渲染（净值），
            仅在旧路径（sh_texture）下生效。

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
    verts, faces, uvs, uv_idx, normals, normal_idx, tangents, bitangents = mesh.to_torch()
    renderer = DifferentiableRenderer(
        vertices=verts,
        faces=faces,
        uvs=uvs,
        uv_idx=uv_idx,
        normals=normals,
        normal_idx=normal_idx,
        tangents=tangents,
        bitangents=bitangents,
        resolution=resolution,
        device=device,
    )

    # Legacy SH path: prepare dc/rest params from sh_texture
    dc_param = None
    rest_param = None
    sub_dc_param = None
    sub_rest_param = None

    if shading_model is None:
        if sh_texture is None:
            raise ValueError("Either sh_texture or shading_model must be provided")
        # sh_texture is the full concatenated texture [1, H, W, n*3]
        # Split into DC + Rest for renderer
        n_dc = 3
        features_dc = sh_texture[..., :n_dc]
        features_rest = sh_texture[..., n_dc:]

        dc_param = nn.Parameter(features_dc.to(device))
        rest_param = nn.Parameter(features_rest.to(device))

        # subtract_texture: 渲染减去该纹理的结果（高频净值）
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
            if shading_model is not None:
                # New path: use ShadingModel
                rast, texc, wpos, interp_normals, vdirs, tang, btan = renderer.rasterize_and_interpolate(cam)
                rgb, mask = shading_model.shade(rast, texc, wpos, interp_normals, vdirs, cam, resolution)
            else:
                # Legacy path: use sh_texture directly
                rgb, mask, _ = renderer.render(dc_param, rest_param, cam)
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


def render_video_multi(
    mesh,
    renderers: dict,
    shading_model,
    submesh_names: list[str],
    output_path: str,
    center=None,
    radius=None,
    height=None,
    fov_deg=None,
    num_frames: int = 120,
    resolution: int = 1024,
    fps: int = 30,
    device: str = "cuda",
    fill_ratio: float = 0.6,
) -> str:
    """Multi-mesh video: render all submeshes per frame, composite.

    Same camera logic as render_video, but loops over submesh renderers
    per frame and composites the results.

    Args:
        mesh: MeshData or MultiMeshData — used for bbox camera auto-calculation.
        renderers: Dict of submesh_name → DifferentiableRenderer.
        shading_model: PBRShadingModel with shade_submesh().
        submesh_names: Ordered list of submesh names.
        output_path: Output mp4 path.
        (remaining args same as render_video)
    """
    # ---- 1. 从 mesh bounding box 自动计算相机参数 ----
    # Get all vertices for bbox calculation
    if hasattr(mesh, 'submeshes'):
        # MultiMeshData — concatenate all submesh vertices
        all_verts = np.concatenate([s.vertices for s in mesh.submeshes], axis=0)
    else:
        all_verts = mesh.vertices

    v_min = all_verts.min(axis=0)
    v_max = all_verts.max(axis=0)
    bbox_size = v_max - v_min
    bbox_max_dim = float(bbox_size.max())

    if center is None:
        center_np = (v_min + v_max) / 2.0
    else:
        center_np = np.array(center, dtype=np.float64)

    dists = np.linalg.norm(all_verts - center_np, axis=1)
    bsphere_radius = float(dists.max())

    if radius is None:
        radius = bsphere_radius * 2.5
    if height is None:
        height = center_np[1] + bsphere_radius * 1.2
    if fov_deg is None:
        cam_y = height
        cam_dist = math.sqrt(radius ** 2 + (cam_y - center_np[1]) ** 2)
        half_angle = math.atan(bsphere_radius / cam_dist)
        fov_deg = math.degrees(2.0 * half_angle / fill_ratio)
        fov_deg = max(20.0, min(fov_deg, 70.0))

    print(f"  [Video Multi] Auto camera: radius={radius:.2f}, height={height:.2f}, fov={fov_deg:.1f}°")

    # ---- 2. 生成轨道相机 ----
    cameras = orbit_cameras(
        center=center_np, radius=radius, height=height,
        num_frames=num_frames, fov_deg=fov_deg, resolution=resolution,
    )

    # ---- 3. 创建视频写入器 ----
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (resolution, resolution))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    # ---- 4. 逐帧渲染 (multi-mesh composite, depth-based) ----
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb = torch.zeros(1, resolution, resolution, 3, device=device)
            depth_buf = torch.full((1, resolution, resolution), float("inf"), device=device)
            mask = torch.zeros(1, resolution, resolution, device=device)

            for sub_name in submesh_names:
                sub_renderer = renderers[sub_name]
                rast, texc, wpos, inorm, vdir, tang, btang = sub_renderer.rasterize_and_interpolate(cam)
                rgb_sub, mask_sub = shading_model.shade_submesh(
                    sub_name, rast, texc, wpos, inorm, vdir, cam, resolution, tang, btang)
                sub_depth = rast[..., 2]
                write = (mask_sub > 0.5) & (sub_depth < depth_buf)
                rgb = torch.where(write.unsqueeze(-1), rgb_sub, rgb)
                depth_buf = torch.where(write, sub_depth, depth_buf)
                mask = torch.max(mask, mask_sub)

        frame = rgb[0].detach().cpu().flip(0).clamp(0.0, 1.0).pow(1.0 / 2.2).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mask_np = mask[0].detach().cpu().flip(0).numpy()
        bg = mask_np < 0.5
        frame[bg] = 0

        writer.write(frame)

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  [Video Multi] Frame {i+1}/{num_frames}")

    writer.release()
    print(f"  [Video Multi] Saved: {output_path}")
    return output_path
