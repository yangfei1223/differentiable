"""球谐（Spherical Harmonics）模块的单元测试。"""

import torch
import pytest

from src.sh import eval_sh_basis, decode_sh, init_sh_texture, _C0


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
# 3. decode_sh — output shape
# ---------------------------------------------------------------------------

def test_decode_sh_output_shape():
    """decode_sh 应返回 [..., 3]"""
    sh = torch.randn(4, 6, 27)
    dirs = torch.randn(4, 6, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors = decode_sh(sh, dirs, order=2)
    assert colors.shape == (4, 6, 3)


# ---------------------------------------------------------------------------
# 4. decode_sh — DC view-independent
# ---------------------------------------------------------------------------

def test_decode_sh_dc_view_independent():
    """仅有 DC 分量时，所有方向应得到相同颜色。"""
    sh = torch.zeros(1, 1, 27)
    dc_color = torch.tensor([0.5, 0.3, 0.8])
    sh[0, 0, :3] = dc_color / _C0  # 反向编码

    dirs = torch.randn(20, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    colors = decode_sh(sh.expand(20, 1, 27), dirs.unsqueeze(1), order=0)
    # colors shape: [20, 1, 3]
    assert torch.allclose(colors[0], colors[1], atol=1e-5)
    assert torch.allclose(colors, colors[0:1], atol=1e-5)


# ---------------------------------------------------------------------------
# 5. init_sh_texture — structure & initial values
# ---------------------------------------------------------------------------

def test_init_sh_texture():
    """init_sh_texture 应生成 [1, H, W, 27]，DC 非零，高阶为零。"""
    res = 16
    sh_order = 2
    init_dc = 0.5

    tex = init_sh_texture(res, sh_order, init_dc=init_dc)

    assert isinstance(tex, torch.nn.Parameter)
    assert tex.shape == (1, res, res, 27)

    # DC 系数 = init_dc / _C0 （对 3 个通道都一样）
    expected_dc = init_dc / _C0
    assert torch.allclose(tex[0, :, :, 0], torch.tensor(expected_dc), atol=1e-6)
    assert torch.allclose(tex[0, :, :, 1], torch.tensor(expected_dc), atol=1e-6)
    assert torch.allclose(tex[0, :, :, 2], torch.tensor(expected_dc), atol=1e-6)

    # 高阶系数应全为零
    assert torch.all(tex[0, :, :, 3:] == 0)
