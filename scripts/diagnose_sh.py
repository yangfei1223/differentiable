"""排查渲染色块问题：SH 各阶系数分析 + 渲染中间过程。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.mesh import load_mesh
from src.renderer import DifferentiableRenderer
from src.camera import load_cameras
from src.sh import eval_sh_basis

mesh = load_mesh("data/scene/lowpoly.glb")
verts, faces, uvs, uv_idx = mesh.to_torch()
cams = load_cameras("data/cameras.json")

ckpt = torch.load("output/epoch1000/sh_texture.pt", map_location="cpu")
tex = ckpt["sh_texture"]  # [1, H, W, 27]
print(f"SH texture: {tex.shape}")

# 1. 各阶系数的值域统计
print("\n=== Per-SH-band statistics ===")
for i in range(9):
    band = tex[0, :, :, i*3:(i+1)*3]
    print(f"SH[{i}]: min={band.min():.4f} max={band.max():.4f} mean={band.mean():.4f} std={band.std():.4f}")

# 2. 渲染一帧，分析中间结果
renderer = DifferentiableRenderer(verts, faces, uvs, uv_idx, resolution=512, device="cuda")
sh_param = nn.Parameter(tex.data.to("cuda"))
cam = cams[0]

with torch.no_grad():
    import nvdiffrast.torch as dr

    mvp = cam.mvp_torch().to("cuda")
    h = w = 512

    v = renderer.vertices
    ones = torch.ones_like(v[..., :1])
    verts_h = torch.cat([v, ones], dim=-1)
    clip = torch.bmm(verts_h, mvp.transpose(1, 2))

    rast, _ = dr.rasterize(renderer.glctx, clip, renderer.faces, resolution=[h, w])
    texc, _ = dr.interpolate(renderer.uvs, rast, renderer.uv_idx)
    world_pos, _ = dr.interpolate(renderer.vertices, rast, renderer.faces)

    cam_pos = torch.tensor(cam.position, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 3)
    view_dir = cam_pos - world_pos
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # SH basis values
    basis = eval_sh_basis(view_dir, order=2)  # [1, H, W, 9]
    print(f"\n=== SH basis values ===")
    for i in range(9):
        b = basis[0, :, :, i]
        valid = b[rast[0, :, :, 3] > 0]
        if len(valid) > 0:
            print(f"  basis[{i}]: min={valid.min():.4f} max={valid.max():.4f} mean={valid.mean():.4f}")

    # Texture sample
    tex_sampled = dr.texture(sh_param, texc, filter_mode="linear", boundary_mode="zero")
    print(f"\n=== Sampled texture ===")
    print(f"  shape: {tex_sampled.shape}")
    valid_mask = rast[0, :, :, 3] > 0
    for i in range(9):
        band = tex_sampled[0, :, :, i*3:(i+1)*3]
        vals = band[valid_mask.unsqueeze(-1).expand_as(band)].reshape(-1, 3)
        if len(vals) > 0:
            print(f"  tex[{i}] RGB: mean=[{vals[:,0].mean():.4f}, {vals[:,1].mean():.4f}, {vals[:,2].mean():.4f}]  max={vals.max():.4f}")

    # Per-band contribution
    sh_9x3 = tex_sampled.reshape(*tex_sampled.shape[:-1], 9, 3)
    print(f"\n=== Per-band contribution (basis * coeff) ===")
    for i in range(9):
        contrib = sh_9x3[0, :, :, i, :] * basis[0, :, :, i:i+1]
        valid_px = contrib[valid_mask]
        if len(valid_px) > 0:
            print(f"  band[{i}]: mean={valid_px.mean():.4f} max={valid_px.max():.4f} min={valid_px.min():.4f}")

    # Final sum per band
    basis_exp = basis.unsqueeze(-1)
    rgb_per_band = (sh_9x3 * basis_exp)  # [1, H, W, 9, 3]
    rgb = rgb_per_band.sum(dim=-2)  # [1, H, W, 3]

    print(f"\n=== Final RGB ===")
    valid_rgb = rgb[0][valid_mask.unsqueeze(-1).expand_as(rgb[0])].reshape(-1, 3)
    if len(valid_rgb) > 0:
        print(f"  mean: [{valid_rgb[:,0].mean():.4f}, {valid_rgb[:,1].mean():.4f}, {valid_rgb[:,2].mean():.4f}]")
        print(f"  min:  [{valid_rgb[:,0].min():.4f}, {valid_rgb[:,1].min():.4f}, {valid_rgb[:,2].min():.4f}]")
        print(f"  max:  [{valid_rgb[:,0].max():.4f}, {valid_rgb[:,1].max():.4f}, {valid_rgb[:,2].max():.4f}]")
        # Negative values?
        neg_count = (valid_rgb < 0).sum()
        print(f"  negative pixels: {neg_count}/{len(valid_rgb)} ({neg_count/len(valid_rgb)*100:.1f}%)")
        # >1 values?
        over_count = (valid_rgb > 1).sum()
        print(f"  >1 pixels: {over_count}/{len(valid_rgb)} ({over_count/len(valid_rgb)*100:.1f}%)")
