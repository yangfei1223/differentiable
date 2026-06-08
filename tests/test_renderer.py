"""可微渲染器的单元测试（需要 CUDA + nvdiffrast）。"""

import torch
import pytest

from src.renderer import DifferentiableRenderer
from src.sh import init_sh_texture, cat_sh_features
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


def _make_sh_params(sh_order=2, init_dc=0.5, resolution=16):
    """创建 DC + Rest 参数对。"""
    _dc, _rest = init_sh_texture(resolution, sh_order=sh_order, init_dc=init_dc)
    return nn.Parameter(_dc.data.cuda()), nn.Parameter(_rest.data.cuda())


import torch.nn as nn


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

    features_dc, features_rest = _make_sh_params()
    rgb, mask, normals = renderer.render(features_dc, features_rest, camera)

    assert rgb.shape == (1, 64, 64, 3), f"rgb shape = {rgb.shape}"
    assert mask.shape == (1, 64, 64), f"mask shape = {mask.shape}"
    assert normals.shape == (1, 64, 64, 3), f"normals shape = {normals.shape}"
    assert mask[0, 32, 32] > 0, "中心像素 mask 应 > 0（quad 可见）"


@cuda_skip
def test_render_gradient_flows():
    """梯度应能从渲染结果流回到 DC 和 Rest 参数。"""
    vertices, faces, uvs, uv_idx = _make_quad_mesh()
    renderer = DifferentiableRenderer(
        vertices, faces, uvs, uv_idx, resolution=64, device="cuda",
    )
    camera = _make_camera()

    features_dc, features_rest = _make_sh_params()
    rgb, mask, _ = renderer.render(features_dc, features_rest, camera)

    loss = rgb.sum()
    loss.backward()

    assert features_dc.grad is not None, "features_dc 应有梯度"
    assert features_dc.grad.abs().sum() > 0, "features_dc 梯度不应全为零"
    assert features_rest.grad is not None, "features_rest 应有梯度"


@cuda_skip
def test_render_dc_color_correct():
    """DC only (order 0) 渲染颜色应 = init_dc（经 +0.5 shift）。"""
    vertices, faces, uvs, uv_idx = _make_quad_mesh()
    renderer = DifferentiableRenderer(
        vertices, faces, uvs, uv_idx, resolution=64, device="cuda",
    )
    camera = _make_camera()

    init_dc = 0.5
    features_dc, features_rest = _make_sh_params(sh_order=0, init_dc=init_dc, resolution=16)
    # order 0: features_rest should be empty [1, H, W, 0]
    assert features_rest.shape[-1] == 0

    rgb, mask, _ = renderer.render(features_dc, features_rest, camera)

    # 3DGS: DC stores (init_dc - 0.5) / C0
    # SH eval: C0 * dc_coeff + 0.5 = C0 * (init_dc-0.5)/C0 + 0.5 = init_dc
    center_rgb = rgb[0, 32, 32]  # [3]
    assert mask[0, 32, 32] > 0, "中心像素应可见"
    assert torch.allclose(center_rgb, torch.tensor(init_dc, device="cuda"), atol=0.05), \
        f"DC color 应接近 {init_dc}, 实际 {center_rgb.tolist()}"


@cuda_skip
def test_renderer_set_uvs():
    """set_uvs should update UVs used for interpolation."""
    import numpy as np

    verts = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.5, 1.0, 0.5]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
    uv_idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    normals = torch.tensor([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]], dtype=torch.float32)
    normal_idx = faces.clone()

    renderer = DifferentiableRenderer(
        vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx,
        normals=normals, normal_idx=normal_idx,
        resolution=64, device="cuda",
    )

    cam = Camera(
        position=np.array([0.5, 0.5, 2.0]),
        look_at=np.array([0.5, 0.5, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0, image_width=64, image_height=64,
    )

    _, texc1, *_ = renderer.rasterize_and_interpolate(cam)

    new_uvs = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], dtype=torch.float32)
    renderer.set_uvs(new_uvs.unsqueeze(0))

    _, texc2, *_ = renderer.rasterize_and_interpolate(cam)

    visible = texc1[..., 0] > 0
    if visible.any():
        diff = (texc1[visible] - texc2[visible]).abs().max().item()
        assert diff > 0.01, "UVs should change after set_uvs"
