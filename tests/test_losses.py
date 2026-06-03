"""损失函数模块的单元测试。"""

import torch
import pytest

from src.losses import l1_loss, ssim_loss, tv_loss, CombinedLoss


# ---------------------------------------------------------------------------
# 1. l1_loss
# ---------------------------------------------------------------------------

def test_l1_zero_on_same():
    """l1_loss(a, a) ≈ 0"""
    a = torch.randn(2, 64, 64, 3)
    loss = l1_loss(a, a)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_l1_nonzero_on_diff():
    """l1_loss(zeros, ones) > 0"""
    zeros = torch.zeros(1, 32, 32, 3)
    ones = torch.ones(1, 32, 32, 3)
    loss = l1_loss(zeros, ones)
    assert loss.item() > 0
    assert loss.item() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. ssim_loss
# ---------------------------------------------------------------------------

def test_ssim_zero_on_same():
    """ssim_loss(a, a) ≈ 0"""
    a = torch.rand(1, 3, 64, 64)
    loss = ssim_loss(a, a)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_ssim_large_on_opposite():
    """ssim_loss(zeros, ones) > 0.5"""
    zeros = torch.zeros(1, 3, 64, 64)
    ones = torch.ones(1, 3, 64, 64)
    loss = ssim_loss(zeros, ones)
    assert loss.item() > 0.5


# ---------------------------------------------------------------------------
# 3. tv_loss
# ---------------------------------------------------------------------------

def test_tv_zero_on_constant():
    """tv_loss(ones) ≈ 0"""
    ones = torch.ones(1, 32, 32, 3)
    loss = tv_loss(ones)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_tv_nonzero_on_noise():
    """tv_loss(randn) > 0"""
    noise = torch.randn(1, 32, 32, 3)
    loss = tv_loss(noise)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# 4. CombinedLoss
# ---------------------------------------------------------------------------

def test_combined_loss_scalar_and_grad():
    """CombinedLoss returns scalar, gradient flows to input."""
    model = CombinedLoss(lambda_l1=1.0, lambda_ssim=1.0, lambda_tv=0.1)

    rendered = torch.rand(1, 32, 32, 3, requires_grad=True)
    gt = torch.rand(1, 32, 32, 3)
    mask = torch.ones(1, 32, 32)
    sh_texture = torch.rand(1, 8, 8, 27, requires_grad=True)

    loss = model(rendered, gt, mask, sh_texture)

    # scalar
    assert loss.dim() == 0
    assert loss.item() >= 0

    # gradient flows to rendered
    loss.backward()
    assert rendered.grad is not None
    assert rendered.grad.abs().sum().item() > 0

    # gradient flows to sh_texture
    assert sh_texture.grad is not None
    assert sh_texture.grad.abs().sum().item() > 0
