"""资产导出模块的单元测试。"""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.exporter import export_diffuse_texture, export_sh_channels, export_gltf
from src.sh import _C0, init_sh_texture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sh_texture_dc(dc: float = 0.5, resolution: int = 16) -> torch.Tensor:
    """创建一个仅含 DC 分量的 SH 纹理 [1, H, W, 27]。"""
    tex = init_sh_texture(resolution, sh_order=2, init_dc=dc)
    return tex.data  # 脱离 nn.Parameter


# ---------------------------------------------------------------------------
# 1. export_diffuse_texture — PNG 输出
# ---------------------------------------------------------------------------

def test_export_diffuse_png():
    """导出 diffuse 纹理为 PNG：验证文件存在且尺寸正确。"""
    resolution = 16
    dc = 0.5
    sh_tex = _make_sh_texture_dc(dc=dc, resolution=resolution)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "diffuse.png")
        export_diffuse_texture(sh_tex, out_path)

        assert os.path.isfile(out_path), "导出的 PNG 文件应存在"

        from PIL import Image
        img = Image.open(out_path)
        assert img.size == (resolution, resolution), (
            f"PNG 尺寸应为 ({resolution}, {resolution})，实际为 {img.size}"
        )

        # 验证像素值接近 dc（DC = sh_dc * _C0 ≈ 0.5）
        arr = np.array(img, dtype=np.float32) / 255.0
        expected = dc  # init_dc=0.5 → DC color = dc_val * _C0 * (1/_C0)... wait
        # sh_tex[0,0,0,0] = dc / _C0, 所以 diffuse[0,0,0] = sh_tex[0,0,0,0] * _C0 = dc
        assert np.allclose(arr.mean(), expected, atol=0.02), (
            f"平均像素值应接近 {expected}，实际为 {arr.mean():.4f}"
        )


# ---------------------------------------------------------------------------
# 2. export_sh_channels — 9 通道 PNG
# ---------------------------------------------------------------------------

def test_export_sh_channels():
    """导出 9 个 SH 通道 PNG：验证文件全部存在。"""
    resolution = 8
    sh_tex = torch.randn(1, resolution, resolution, 27)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_sh_channels(sh_tex, tmpdir)

        assert len(paths) == 9, f"应返回 9 个路径，实际 {len(paths)}"
        for p in paths:
            assert os.path.isfile(p), f"通道文件不存在: {p}"

        # 验证命名格式: sh_00.png, sh_01.png, ..., sh_08.png
        for i, p in enumerate(paths):
            expected_name = f"sh_{i:02d}.png"
            assert os.path.basename(p) == expected_name, (
                f"第 {i} 个文件名应为 {expected_name}，实际为 {os.path.basename(p)}"
            )


# ---------------------------------------------------------------------------
# 3. export_gltf — .glb 导出
# ---------------------------------------------------------------------------

def test_export_gltf():
    """导出简单 quad mesh + SH 纹理为 .glb 文件。"""
    # 简单 quad: 两个三角形
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)

    resolution = 4
    sh_tex = _make_sh_texture_dc(dc=0.6, resolution=resolution)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "model.glb")
        export_gltf(vertices, faces, sh_tex, out_path)

        assert os.path.isfile(out_path), "导出的 .glb 文件应存在"
        assert os.path.getsize(out_path) > 0, ".glb 文件不应为空"
