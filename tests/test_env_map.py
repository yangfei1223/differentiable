"""测试 equirect 环境贴图。"""
import torch
from src.shading.pbr.env_map import (
    init_env_map,
    direction_to_equirect,
    sample_env_map,
)


def test_init_env_map_default():
    env = init_env_map(32, 64)
    assert env.shape == (1, 32, 64, 3)
    assert isinstance(env, torch.nn.Parameter)


def test_direction_to_equirect_forward():
    """正前方 (+Z) 应映射到 equirect 中心。"""
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    u, v = direction_to_equirect(dirs)
    assert abs(u[0].item() - 0.75) < 0.01
    assert abs(v[0].item() - 0.5) < 0.01


def test_direction_to_equirect_up():
    """正上方 (+Y) 应映射到 equirect 顶部。"""
    dirs = torch.tensor([[0.0, 1.0, 0.0]])
    u, v = direction_to_equirect(dirs)
    # clamp(-0.999, 0.999) causes slight deviation from 1.0
    assert abs(v[0].item() - 1.0) < 0.02


def test_direction_to_equirect_gradient():
    dirs = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    u, v = direction_to_equirect(dirs)
    u.sum().backward()
    assert dirs.grad is not None


def test_sample_env_map_basic():
    """无 mip level 时直接采样（linear）。"""
    env = init_env_map(16, 32).cuda()
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    color = sample_env_map(env, dirs)
    assert color.shape == (1, 3)


def test_sample_env_map_with_mip():
    """带 mip_level_bias 时使用 nvdiffrast mipmap。"""
    env = init_env_map(32, 64).cuda()
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    roughness = torch.tensor([[0.5]]).cuda()
    color = sample_env_map(env, dirs, mip_level_bias=roughness)
    assert color.shape == (1, 3)


def test_sample_env_map_gradient():
    """采样可导。"""
    import torch.nn as nn
    raw = init_env_map(16, 32)
    env = nn.Parameter(raw.data.cuda())  # 确保是 leaf tensor
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    color = sample_env_map(env, dirs)
    loss = color.sum()
    loss.backward()
    assert env.grad is not None
