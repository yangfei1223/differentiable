"""生成 GT vs Rendered 对比图（gamma 校正后）。"""
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

for i in range(3):
    cam = cams[i]

    with torch.no_grad():
        rgb, mask = renderer.render(sh_param, cam)
    # linear → sRGB gamma
    render_np = rgb[0].cpu().flip(0).clamp(0, 1).pow(1.0/2.2).numpy()
    render_np = (render_np * 255).astype(np.uint8)
    render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

    # Mask out background
    mask_np = mask[0].cpu().flip(0).numpy()
    render_bgr[mask_np < 0.5] = 0

    gt = cv2.imread(f"data/gt/view_{i:04d}.png")

    h1, w1 = render_bgr.shape[:2]
    h2, w2 = gt.shape[:2]
    target_h = min(h1, h2)
    r1 = cv2.resize(render_bgr, (w1 * target_h // h1, target_h))
    g1 = cv2.resize(gt, (w2 * target_h // h2, target_h))

    # Add label
    cv2.putText(r1, "Rendered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(g1, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    canvas = np.concatenate([g1, r1], axis=1)
    cv2.imwrite(f"output/compare_{i:04d}.png", canvas)

    # Stats on object pixels only
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_RGB2BGR).astype(float) / 255.0
    gt_obj = gt_rgb[gt_rgb.max(axis=2) > 0.01]
    render_obj = render_rgb[mask_np > 0.5]
    if len(gt_obj) > 0 and len(render_obj) > 0:
        print(f"view_{i:04d}:")
        print(f"  GT      mean: [{gt_obj[:,0].mean():.3f}, {gt_obj[:,1].mean():.3f}, {gt_obj[:,2].mean():.3f}]")
        print(f"  Render  mean: [{render_obj[:,0].mean():.3f}, {render_obj[:,1].mean():.3f}, {render_obj[:,2].mean():.3f}]")
        print(f"  GT      max: {gt_obj.max():.3f}")
        print(f"  Render  max: {render_obj.max():.3f}")

print("\nSaved output/compare_0*.png")
