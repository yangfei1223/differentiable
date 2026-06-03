"""生成 GT vs Rendered 对比图，诊断视角/颜色是否匹配。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.mesh import load_mesh
from src.renderer import DifferentiableRenderer
from src.camera import load_cameras

mesh = load_mesh("data/scene/lowpoly.glb")
verts, faces, uvs, uv_idx = mesh.to_torch()
cams = load_cameras("data/cameras.json")

ckpt = torch.load("output/sh_texture_epoch2000.pt", map_location="cpu")
tex = ckpt["sh_texture"]

renderer = DifferentiableRenderer(verts, faces, uvs, uv_idx, resolution=1024, device="cuda")
sh_param = nn.Parameter(tex.data.to("cuda"))

# Render first 3 views
for i in range(3):
    cam = cams[i]

    # Render
    with torch.no_grad():
        rgb, mask = renderer.render(sh_param, cam)
    render_np = rgb[0].cpu().flip(0).clamp(0, 1).numpy()
    render_np = (render_np * 255).astype(np.uint8)
    render_np = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

    # GT
    gt = cv2.imread(f"data/gt/view_{i:04d}.png")

    # Side by side
    h1, w1 = render_np.shape[:2]
    h2, w2 = gt.shape[:2]
    target_h = min(h1, h2)
    render_np = cv2.resize(render_np, (w1 * target_h // h1, target_h))
    gt = cv2.resize(gt, (w2 * target_h // h2, target_h))
    canvas = np.concatenate([gt, render_np], axis=1)

    cv2.imwrite(f"output/compare_{i:04d}.png", canvas)
    print(f"Saved compare_{i:04d}.png  (GT left, Rendered right)")

    # Stats
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    render_rgb = cv2.cvtColor(render_np, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    gt_obj = gt_rgb[gt_rgb.max(axis=2) > 0.01]
    render_obj = render_rgb[mask[0].cpu().flip(0).numpy() > 0.5]
    if len(gt_obj) > 0 and len(render_obj) > 0:
        print(f"  GT obj mean: [{gt_obj[:,0].mean():.3f}, {gt_obj[:,1].mean():.3f}, {gt_obj[:,2].mean():.3f}]")
        print(f"  Render obj mean: [{render_obj[:,0].mean():.3f}, {render_obj[:,1].mean():.3f}, {render_obj[:,2].mean():.3f}]")

print("\nDone. Check output/compare_*.png")
