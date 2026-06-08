import torch
import numpy as np


def test_uv_optimizer_step():
    """UVOptimizer.step should reduce loss."""
    from src.uv.param import UVParameterizer
    from src.uv.optimizer import UVOptimizer

    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    optimizer = UVOptimizer(param, lr=0.1, max_iter=5)

    def closure():
        optimizer.zero_grad()
        decoded = param.get_uvs()
        loss = ((decoded - 0.7) ** 2).sum()
        loss.backward()
        return loss

    loss_before = closure().item()
    optimizer.step(closure)
    loss_after = closure().item()
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"


def test_uv_optimizer_zero_grad():
    """zero_grad should clear gradients."""
    from src.uv.param import UVParameterizer
    from src.uv.optimizer import UVOptimizer

    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    optimizer = UVOptimizer(param, lr=0.1, max_iter=5)

    decoded = param.get_uvs()
    decoded.sum().backward()
    assert param.raw.grad is not None

    optimizer.zero_grad()
    assert param.raw.grad is None or param.raw.grad.norm().item() == 0
