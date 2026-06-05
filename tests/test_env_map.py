"""测试 equirect 环境贴图。"""
import torch
from src.shading.pbr.env_map import (
    init_env_map,
    direction_to_equirect,
    sample_env_map,
    prefilter_env_map,
    sample_prefiltered,
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


def test_sample_env_map():
    env = init_env_map(16, 32)
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    color = sample_env_map(env, dirs)
    assert color.shape == (1, 3)


def test_prefilter_env_map_shape():
    env = init_env_map(16, 32)
    n_levels = 5
    prefiltered = prefilter_env_map(env, n_levels=n_levels)
    assert prefiltered.shape[0] == 1
    assert prefiltered.shape[1] == n_levels
    assert prefiltered.shape[2] == 16
    assert prefiltered.shape[3] == 32
    assert prefiltered.shape[4] == 3


def test_prefilter_env_map_gradient():
    env = init_env_map(16, 32)
    prefiltered = prefilter_env_map(env, n_levels=3)
    loss = prefiltered.sum()
    loss.backward()
    assert env.grad is not None


def test_sample_prefiltered():
    env = init_env_map(16, 32)
    prefiltered = prefilter_env_map(env, n_levels=5)
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    roughness = torch.tensor([[0.5]])
    color = sample_prefiltered(prefiltered, dirs, roughness, n_levels=5)
    assert color.shape == (1, 3)
