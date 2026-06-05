"""诊断渲染质量。"""
import cv2
import numpy as np
import torch

# 1. GT 图色彩分布
print("=== GT Images ===")
for i in [0, 50, 100, 150]:
    img = cv2.imread(f"data/gt/view_{i:04d}.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    non_black = img_rgb[img_rgb.max(axis=2) > 0.01]
    if len(non_black) > 0:
        print(f"  view_{i:04d}: mean_rgb=[{non_black[:,0].mean():.3f}, {non_black[:,1].mean():.3f}, {non_black[:,2].mean():.3f}]")
        print(f"            max={img_rgb.max():.3f}, nonblack_px={len(non_black)}, total_px={img_rgb.shape[0]*img_rgb.shape[1]}")
    else:
        print(f"  view_{i:04d}: ALL BLACK")

# 2. 渲染一帧，对比 GT
print("\n=== Single Frame Render vs GT ===")
from src.mesh import load_mesh
from src.renderer import DifferentiableRenderer
from src.camera import load_cameras
import torch.nn as nn

mesh = load_mesh("data/scene/lowpoly.glb")
verts, faces, uvs, uv_idx = mesh.to_torch()
cams = load_cameras("data/cameras.json")

ckpt = torch.load("output/sh_texture_epoch2000.pt", map_location="cpu")
tex = ckpt["sh_texture"]
print(f"Texture shape: {tex.shape}")

renderer = DifferentiableRenderer(verts, faces, uvs, uv_idx, resolution=512, device="cuda")
sh_param = nn.Parameter(tex.data.to("cuda"))

with torch.no_grad():
    rgb, mask, _ = renderer.render(sh_param, cams[0])

rgb_np = rgb[0].cpu().clamp(0, 1).numpy()
mask_np = mask[0].cpu().numpy()

print(f"Rendered: shape={rgb_np.shape}, range=[{rgb_np.min():.4f}, {rgb_np.max():.4f}]")
rendered_pixels = rgb_np[mask_np > 0.5]
if len(rendered_pixels) > 0:
    print(f"  Non-masked pixels mean RGB: [{rendered_pixels[:,0].mean():.4f}, {rendered_pixels[:,1].mean():.4f}, {rendered_pixels[:,2].mean():.4f}]")
    print(f"  Non-masked pixels max: {rendered_pixels.max():.4f}")
else:
    print("  ALL MASKED OUT")

print(f"Mask coverage: {(mask_np > 0.5).sum()}/{mask_np.size} ({(mask_np > 0.5).mean()*100:.1f}%)")

# 3. Check GT at render resolution
gt_img = cv2.imread("data/gt/view_0000.png")
gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
gt_resized = cv2.resize(gt_img, (512, 512))
gt_mask = gt_img.max(axis=2) > 0.01
gt_nonblack = gt_img[gt_mask]
print(f"\nGT view_0000: shape={gt_img.shape}, nonblack={len(gt_nonblack)}")
if len(gt_nonblack) > 0:
    print(f"  GT nonblack mean RGB: [{gt_nonblack[:,0].mean():.3f}, {gt_nonblack[:,1].mean():.3f}, {gt_nonblack[:,2].mean():.3f}]")

# 4. 检查 UV 覆盖 — 渲染时 UV 坐标是否落在纹理内
print("\n=== UV Coverage ===")
u = uvs.numpy()[:, 0]
v = uvs.numpy()[:, 1]
print(f"U range: [{u.min():.4f}, {u.max():.4f}]")
print(f"V range: [{v.min():.4f}, {v.max():.4f}]")
# nvdiffrast boundary_mode='zero' means out-of-range UVs return black
out_of_range = ((u < 0) | (u > 1) | (v < 0) | (v > 1)).sum()
print(f"Out-of-range UVs: {out_of_range}/{len(u)} ({out_of_range/len(u)*100:.2f}%)")
