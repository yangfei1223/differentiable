"""资产导出：将 SH 纹理导出为 Diffuse PNG、多通道 SH PNG、或 glTF (.glb)。

依赖：numpy, Pillow, trimesh（仅 glTF 导出）。
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image

from src.sh import _C0, SH2RGB


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """将 float32 张量 [1, H, W, C]（值域 [0,1]）转为 uint8 numpy 数组。

    Args:
        tensor: 形状 ``[1, H, W, C]``，值域 ``[0, 1]``。

    Returns:
        numpy 数组，形状 ``[H, W, C]``，dtype ``uint8``。
    """
    # 移除 batch 维度 → [H, W, C]
    arr = tensor[0].detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# export_diffuse_texture
# ---------------------------------------------------------------------------

def export_diffuse_texture(
    sh_texture: torch.Tensor,
    output_path: str,
    sh_order: int = 2,
) -> str:
    """从 SH 纹理提取 DC 分量（漫反射颜色）并保存为 PNG。

    DC 颜色使用 3DGS 约定: SH2RGB(dc_coeff) = dc_coeff * C0 + 0.5。

    Args:
        sh_texture: SH 系数，形状 ``[1, H, W, 27]``。
        output_path: 输出 PNG 文件路径。
        sh_order: SH 阶数（用于确定使用多少系数，但 DC 始终为前 3 个通道）。

    Returns:
        保存的文件路径。
    """
    # DC 系数在前 3 个通道: sh_texture[..., 0] 是 R-DC, ..., sh_texture[..., 2] 是 B-DC
    dc_coeffs = sh_texture[..., :3]  # [1, H, W, 3]

    # DC 颜色 = SH2RGB(dc_coeff) = dc_coeff * C0 + 0.5
    diffuse = SH2RGB(dc_coeffs)  # [1, H, W, 3]

    # 线性空间 → sRGB gamma 校正
    diffuse = diffuse.clamp(0, 1).pow(1.0 / 2.2)

    # Clamp 到 [0, 1] 并转 uint8
    uint8_img = _to_uint8(diffuse)  # [H, W, 3]

    img = Image.fromarray(uint8_img, mode="RGB")
    img.save(output_path)

    return output_path


# ---------------------------------------------------------------------------
# export_sh_channels
# ---------------------------------------------------------------------------

def export_sh_channels(
    sh_texture: torch.Tensor,
    output_dir: str,
    sh_order: int = 2,
) -> List[str]:
    """将 SH 纹理按 9 个基函数拆分为 9 张 PNG，每张包含 R/G/B 三通道。

    Args:
        sh_texture: SH 系数，形状 ``[1, H, W, 27]``（9 基函数 × 3 颜色通道）。
        output_dir: 输出目录。
        sh_order: SH 阶数。

    Returns:
        9 个 PNG 文件路径的列表。
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reshape 为 [1, H, W, 9, 3]
    reshaped = sh_texture.reshape(*sh_texture.shape[:-1], 9, 3)

    paths: List[str] = []
    for i in range(9):
        channel_i = reshaped[..., i, :]  # [1, H, W, 3]

        # 将 SH 系数归一化到可视化范围：简单 clamp 到合理值后归一化
        # 为了可视化，直接 clamp 到 [-max, max] → [0, 1]
        abs_max = channel_i.abs().max().clamp(min=1e-6)
        normalized = (channel_i / abs_max + 1.0) / 2.0  # 映射到 [0, 1]

        uint8_img = _to_uint8(normalized)
        img = Image.fromarray(uint8_img, mode="RGB")

        filename = f"sh_{i:02d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        paths.append(filepath)

    return paths


# ---------------------------------------------------------------------------
# export_gltf
# ---------------------------------------------------------------------------

def export_gltf(
    vertices: np.ndarray,
    faces: np.ndarray,
    sh_texture: torch.Tensor,
    output_path: str,
) -> str:
    """导出 mesh + diffuse 纹理为 glTF Binary (.glb)。

    Args:
        vertices: 顶点坐标，形状 ``[N, 3]``。
        faces: 三角形面片索引，形状 ``[F, 3]``。
        sh_texture: SH 纹理，形状 ``[1, H, W, 27]``。
        output_path: 输出 .glb 文件路径。

    Returns:
        保存的文件路径。
    """
    import trimesh

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 创建 trimesh mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    # 导出 diffuse 纹理为临时 PNG
    output_dir = os.path.dirname(os.path.abspath(output_path))
    tex_path = os.path.join(output_dir, "_diffuse_tmp.png")
    export_diffuse_texture(sh_texture, tex_path)

    # 为 mesh 添加纹理
    from PIL import Image as PILImage

    img = PILImage.open(tex_path)

    # 创建简单的 UV 坐标（覆盖整个纹理）
    n_verts = len(vertices)
    uvs = np.zeros((n_verts, 2), dtype=np.float32)
    # 简单映射: 基于 x, y 坐标
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    extent = v_max - v_min
    extent[extent == 0] = 1.0  # 避免除以零
    uvs[:, 0] = (vertices[:, 0] - v_min[0]) / extent[0]
    uvs[:, 1] = (vertices[:, 1] - v_min[1]) / extent[1]

    # 将纹理附加到 mesh
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    texture_visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=img, material=material)
    mesh.visual = texture_visual

    # 导出为 .glb
    glb_data = mesh.export(file_type="glb")
    with open(output_path, "wb") as f:
        f.write(glb_data)

    # 清理临时纹理文件
    if os.path.exists(tex_path):
        os.remove(tex_path)

    return output_path
