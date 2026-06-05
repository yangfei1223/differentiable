"""球谐（Spherical Harmonics）模块的单元测试。"""

import torch
import pytest

from src.sh import eval_sh_basis, decode_sh, init_sh_texture, _C0, RGB2SH, SH2RGB, cat_sh_features


# ---------------------------------------------------------------------------
# 1. eval_sh_basis — shape tests
# ---------------------------------------------------------------------------

def test_sh_basis_order0_shape():
    """order=0 → [..., 1]"""
    dirs = torch.randn(10, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    basis = eval_sh_basis(dirs, order=0)
    assert basis.shape == (10, 1)


def test_sh_basis_order2_shape():
    """order=2 → [..., 9]  (1 + 3 + 5)"""
    dirs = torch.randn(5, 8, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    basis = eval_sh_basis(dirs, order=2)
    assert basis.shape == (5, 8, 9)


# ---------------------------------------------------------------------------
# 2. eval_sh_basis — DC constant
# ---------------------------------------------------------------------------

def test_sh_dc_constant():
    """DC 系数对所有方向均为常数 0.28209…"""
    dirs = torch.randn(100, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    basis = eval_sh_basis(dirs, order=0)
    dc_vals = basis[..., 0]
    expected = torch.full_like(dc_vals, 0.28209479177387814)
    assert torch.allclose(dc_vals, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. RGB2SH / SH2RGB roundtrip
# ---------------------------------------------------------------------------

def test_rgb_sh_roundtrip():
    """RGB2SH → SH2RGB 应还原原始值。"""
    rgb = torch.tensor([0.0, 0.5, 1.0])
    sh = RGB2SH(rgb)
    rgb_back = SH2RGB(sh)
    assert torch.allclose(rgb, rgb_back, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. decode_sh — output shape
# ---------------------------------------------------------------------------

def test_decode_sh_output_shape():
    """decode_sh 应返回 [..., 3]"""
    sh = torch.randn(4, 6, 27)
    dirs = torch.randn(4, 6, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors = decode_sh(sh, dirs, order=2)
    assert colors.shape == (4, 6, 3)


# ---------------------------------------------------------------------------
# 5. decode_sh — DC view-independent (raw SH output, no +0.5 shift)
# ---------------------------------------------------------------------------

def test_decode_sh_dc_view_independent():
    """仅有 DC 分量时，所有方向应得到相同颜色。"""
    sh = torch.zeros(1, 1, 27)
    dc_color = torch.tensor([0.5, 0.3, 0.8])
    # 3DGS 约定: DC 存储 RGB2SH(color) = (color - 0.5) / C0
    sh[0, 0, :3] = RGB2SH(dc_color)

    dirs = torch.randn(20, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # decode_sh 返回原始 SH 加权和（不含 +0.5）
    colors = decode_sh(sh.expand(20, 1, 27), dirs.unsqueeze(1), order=0)
    # 所有方向应相同
    assert torch.allclose(colors, colors[0:1], atol=1e-5)

    # 原始 SH 输出 = C0 * RGB2SH(dc_color) = dc_color - 0.5
    expected_raw = dc_color - 0.5
    assert torch.allclose(colors[0, 0], expected_raw, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. init_sh_texture — structure & initial values (3DGS style)
# ---------------------------------------------------------------------------

def test_init_sh_texture():
    """init_sh_texture 应返回 (features_dc, features_rest)。"""
    res = 16
    sh_order = 2
    init_dc = 0.5

    features_dc, features_rest = init_sh_texture(res, sh_order, init_dc=init_dc)

    assert isinstance(features_dc, torch.nn.Parameter)
    assert isinstance(features_rest, torch.nn.Parameter)
    assert features_dc.shape == (1, res, res, 3)
    assert features_rest.shape == (1, res, res, 24)  # (9-1)*3 = 24

    # DC 系数 = RGB2SH(init_dc) = (init_dc - 0.5) / C0
    expected_dc = RGB2SH(init_dc)
    assert torch.allclose(features_dc[0, :, :, 0], torch.tensor(expected_dc), atol=1e-6)
    assert torch.allclose(features_dc[0, :, :, 1], torch.tensor(expected_dc), atol=1e-6)
    assert torch.allclose(features_dc[0, :, :, 2], torch.tensor(expected_dc), atol=1e-6)

    # 高阶系数应全为零
    assert torch.all(features_rest == 0)


# ---------------------------------------------------------------------------
# 7. cat_sh_features — concatenation
# ---------------------------------------------------------------------------

def test_cat_sh_features():
    """cat_sh_features 应正确拼接 DC 和 Rest。"""
    dc = torch.randn(1, 8, 8, 3)
    rest = torch.randn(1, 8, 8, 24)
    full = cat_sh_features(dc, rest)
    assert full.shape == (1, 8, 8, 27)
    assert torch.allclose(full[..., :3], dc)
    assert torch.allclose(full[..., 3:], rest)
