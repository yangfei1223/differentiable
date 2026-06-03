"""可微渲染器的单元测试（需要 CUDA + nvdiffrast）。"""

import torch
import pytest

from src.renderer import DifferentiableRenderer
from src.sh import init_sh_texture
from src.camera import Camera


# ---------------------------------------------------------------------------
# CUDA 跳过标记
# ---------------------------------------------------------------------------
cuda_skip = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA",
)


def _make_quad_mesh():
    """创建一个中心在原点的 XY 平面四边形，返回 (vertices, faces, uvs, uv_idx)。"""
    device = "cuda"

    vertices = torch.tensor(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3]],
        dtype=torch.int32,
        device=device,
    )
    uvs = torch.tensor(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=torch.float32,
        device=device,
    )
    uv_idx = faces.clone()

    return vertices, faces, uvs, uv_idx


def _make_camera():
    """创建一个正对 quad 的相机。"""
    import numpy as np

    return Camera(
        position=np.array([0.0, 0.0, 3.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        image_width=64,
        image_height=64,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cuda_skip
def test_render_output_shape():
    """渲染结果形状：rgb [1,64,64,3]，mask [1,64,64]，中心像素 mask > 0。"""
    vertices, faces, uvs, uv_idx = _make_quad_mesh()
    renderer = DifferentiableRenderer(
        vertices, faces, uvs, uv_idx, resolution=64, device="cuda",
    )
    camera = _make_camera()

    sh_texture = torch.nn.Parameter(init_sh_texture(16, sh_order=2, init_dc=0.5).data.cuda())
    rgb, mask = renderer.render(sh_texture, camera)

    assert rgb.shape == (1, 64, 64, 3), f"rgb shape = {rgb.shape}"
    assert mask.shape == (1, 64, 64), f"mask shape = {mask.shape}"
    assert mask[0, 32, 32] > 0, "中心像素 mask 应 > 0（quad 可见）"


@cuda_skip
def test_render_gradient_flows():
    """梯度应能从渲染结果流回到 sh_texture 参数。"""
    vertices, faces, uvs, uv_idx = _make_quad_mesh()
    renderer = DifferentiableRenderer(
        vertices, faces, uvs, uv_idx, resolution=64, device="cuda",
    )
    camera = _make_camera()

    sh_texture = torch.nn.Parameter(init_sh_texture(16, sh_order=2, init_dc=0.5).data.cuda())
    rgb, mask = renderer.render(sh_texture, camera)

    loss = rgb.sum()
    loss.backward()

    assert sh_texture.grad is not None, "sh_texture 应有梯度"
    assert sh_texture.grad.abs().sum() > 0, "sh_texture 梯度不应全为零"
