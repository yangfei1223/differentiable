"""测试 EnvironmentMap 类。"""
import torch
import torch.nn as nn
from src.shading.pbr.env_map import EnvironmentMap


def test_env_map_init_default():
    env = EnvironmentMap(32, 64)
    assert env.raw.shape == (1, 32, 64, 3)
    assert isinstance(env.raw, nn.Parameter)


def test_env_map_init_with_image():
    img = torch.rand(1, 16, 32, 3) * 2.0  # HDR values
    env = EnvironmentMap(16, 32, init_image=img)
    decoded = env.decode()
    assert decoded.shape == (1, 16, 32, 3)
    # softplus(x) > 0 for all x, decoded should be close to original
    assert (decoded > 0).all()


def test_env_map_decode_positive():
    env = EnvironmentMap(16, 32)
    decoded = env.decode()
    assert (decoded > 0).all()


def test_env_map_direction_to_uv():
    uv = EnvironmentMap.direction_to_uv(torch.tensor([[0.0, 0.0, 1.0]]))
    assert abs(uv[0, 0].item() - 0.75) < 0.01
    assert abs(uv[0, 1].item() - 0.5) < 0.01


def test_env_map_direction_to_uv_up():
    uv = EnvironmentMap.direction_to_uv(torch.tensor([[0.0, 1.0, 0.0]]))
    assert abs(uv[0, 1].item() - 1.0) < 0.02


def test_env_map_direction_to_uv_gradient():
    dirs = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    uv = EnvironmentMap.direction_to_uv(dirs)
    uv.sum().backward()
    assert dirs.grad is not None




def test_env_map_sample_basic():
    env = EnvironmentMap(32, 64).cuda()
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    color = env.sample(dirs)
    assert color.shape == (1, 3)


def test_env_map_sample_with_mip():
    env = EnvironmentMap(32, 64).cuda()
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    mip_level = torch.tensor([[0.5]]).cuda()
    color = env.sample(dirs, mip_level=mip_level)
    assert color.shape == (1, 3)


def test_env_map_sample_diffuse():
    env = EnvironmentMap(32, 64).cuda()
    normals = torch.tensor([[0.0, 1.0, 0.0]]).cuda()
    color = env.sample_diffuse(normals)
    assert color.shape == (1, 3)


def test_env_map_sample_specular():
    env = EnvironmentMap(32, 64).cuda()
    reflect = torch.tensor([[0.0, 1.0, 0.0]]).cuda()
    roughness = torch.tensor([[0.5]]).cuda()
    color = env.sample_specular(reflect, roughness)
    assert color.shape == (1, 3)


def test_env_map_gradient():
    env = EnvironmentMap(16, 32).cuda()
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    color = env.sample(dirs)
    loss = color.sum()
    loss.backward()
    assert env.raw.grad is not None


def test_env_map_mipmap_gradient():
    env = EnvironmentMap(16, 32).cuda()
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    mip_level = torch.tensor([[0.5]]).cuda()
    color = env.sample(dirs, mip_level=mip_level)
    loss = color.sum()
    loss.backward()
    assert env.raw.grad is not None
