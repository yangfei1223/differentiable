import torch
import numpy as np


def test_uv_param_sigmoid_roundtrip():
    """Sigmoid encode → decode should recover original UVs."""
    from src.uv.param import UVParameterizer
    uvs = np.random.rand(100, 2).astype(np.float32) * 0.8 + 0.1
    uv_idx = np.zeros((50, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    decoded = param.get_uvs()
    assert decoded.shape == (100, 2)
    diff = (decoded - torch.from_numpy(uvs)).abs().max().item()
    assert diff < 0.02, f"Roundtrip error too large: {diff}"


def test_uv_param_requires_grad():
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    assert param.raw.requires_grad is True


def test_uv_param_get_uv_idx():
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    idx = param.get_uv_idx()
    assert idx.dtype == torch.int32
    assert (idx == torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)).all()


def test_uv_param_gradient_flows():
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    decoded = param.get_uvs()
    loss = decoded.sum()
    loss.backward()
    assert param.raw.grad is not None
    assert param.raw.grad.norm().item() > 0


def test_uv_param_output_range():
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    with torch.no_grad():
        param.raw.fill_(10.0)
    decoded = param.get_uvs()
    assert decoded.min().item() > 0.0
    assert decoded.max().item() < 1.0
    with torch.no_grad():
        param.raw.fill_(-10.0)
    decoded = param.get_uvs()
    assert decoded.min().item() > 0.0
    assert decoded.max().item() < 1.0
