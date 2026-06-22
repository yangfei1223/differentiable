"""Standalone Python PBR specular test.

Reproduces Python's env_map.sample_specular() using only numpy + torch,
to compare with WebGL shader output.

We sample env_map at the same reflection directions that the helmet's
visible pixels would produce. To approximate, we use the same camera,
sample the env map at uniformly distributed reflection directions, and
see if the resulting "specular" is neutral or warm.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys, os, json
sys.path.insert(0, '.')

# Load decoded env
state = torch.load('output/helmet_260604_pbr/epoch2000/pbr_checkpoint.pt', map_location='cpu', weights_only=False)
raw = state['env_map']
env = F.softplus(raw)[0]  # [256, 512, 3]

# Generate mip chain manually (box filter, like Python's nvdiffrast)
mips = [env]
cur = env
while cur.shape[0] > 1 and cur.shape[1] > 1:
    cur_t = cur.permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(cur_t, 2)[0].permute(1, 2, 0)
    mips.append(pooled)
    cur = pooled
print(f'Generated {len(mips)} mip levels')
for i, m in enumerate(mips):
    print(f'  mip {i}: shape={m.shape}, mean={m.mean(dim=(0,1)).tolist()}')

# Test: generate random reflection directions (uniform on sphere)
# and sample env at mip 5 (matches Python's specular mip for roughness=0.55)
np.random.seed(42)
N_samples = 10000
# Uniform sphere: phi = 2*pi*u, theta = acos(2v-1)
u1, u2 = np.random.rand(N_samples), np.random.rand(N_samples)
phi = 2 * np.pi * u1
cos_theta = 2 * u2 - 1
sin_theta = np.sqrt(1 - cos_theta**2)
dirs = np.stack([sin_theta * np.cos(phi), cos_theta, sin_theta * np.sin(phi)], axis=-1)

# Convert to UV (matches env_map.py direction_to_uv)
def direction_to_uv(dirs):
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    u = np.arctan2(z, x) / (2.0 * np.pi) + 0.5
    v = np.arcsin(np.clip(y, -0.999, 0.999)) / np.pi + 0.5
    return np.stack([u, v], axis=-1)

uvs = direction_to_uv(dirs)

# Sample at mip 5 (8x16) with bilinear
def sample_env(env_mip, uvs):
    H, W = env_mip.shape[:2]
    # Bilinear with wrap
    fx = uvs[:, 0] * W - 0.5
    fy = uvs[:, 1] * H - 0.5
    x0 = np.floor(fx).astype(int) % W
    x1 = (x0 + 1) % W
    y0 = np.floor(fy).astype(int) % H
    y1 = (y0 + 1) % H
    wx = fx - np.floor(fx)
    wy = fy - np.floor(fy)
    env_np = env_mip.numpy() if isinstance(env_mip, torch.Tensor) else env_mip
    c00 = env_np[y0, x0]
    c01 = env_np[y1, x0]
    c10 = env_np[y0, x1]
    c11 = env_np[y1, x1]
    out = (1-wx[:, None])*(1-wy[:, None])*c00 + (1-wx[:, None])*wy[:, None]*c01 + \
          wx[:, None]*(1-wy[:, None])*c10 + wx[:, None]*wy[:, None]*c11
    return out

# Sample at multiple mip levels
for mip_idx in [0, 3, 5, 7, 9]:
    if mip_idx >= len(mips): continue
    sampled = sample_env(mips[mip_idx], uvs)
    print(f'mip {mip_idx}: mean sampled = {sampled.mean(axis=0).tolist()}, R/G={sampled[:, 0].mean()/sampled[:, 1].mean():.3f}, B/G={sampled[:, 2].mean()/sampled[:, 1].mean():.3f}')

# Now sample at fractional mip (trilinear between mip 5 and mip 6)
def sample_trilinear(uvs, mip_level):
    lower = int(np.floor(mip_level))
    upper = min(lower + 1, len(mips) - 1)
    frac = mip_level - lower
    s0 = sample_env(mips[lower], uvs)
    s1 = sample_env(mips[upper], uvs)
    return s0 * (1 - frac) + s1 * frac

print('\nTrilinear sampling (roughness * 9):')
for r in [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]:
    mip = r * 9
    sampled = sample_trilinear(uvs, mip)
    print(f'  r={r}: mip={mip:.1f}, mean={sampled.mean(axis=0).tolist()}, B/G={sampled[:, 2].mean()/sampled[:, 1].mean():.3f}')
