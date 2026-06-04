"""快速诊断：SH order 0 vs order 2 对比 — 确认问题来源。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from src.config import load_config
from src.mesh import load_mesh
from src.dataset import GTDataset
from src.renderer import DifferentiableRenderer
from src.losses import CombinedLoss
from src.sh import init_sh_texture

cfg = load_config("configs/train_1k.yaml")
device = "cuda"
mesh = load_mesh(cfg.data.mesh_path)
verts, faces, uvs, uv_idx = mesh.to_torch()
ds = GTDataset(gt_dir=cfg.data.gt_dir, camera_path=cfg.data.camera_path)

# ---- Test 1: SH order 0 (diffuse only) ----
print("=== SH Order 0 (Diffuse only) ===")
sh0 = init_sh_texture(512, sh_order=0, init_dc=0.5)
sh0_param = nn.Parameter(sh0.data.to(device))
renderer0 = DifferentiableRenderer(verts, faces, uvs, uv_idx, resolution=512, device=device)

optimizer0 = torch.optim.Adam([sh0_param], lr=0.01)
criterion = CombinedLoss(lambda_l1=1.0, lambda_ssim=0.2, lambda_tv=0.005)

for step in range(200):
    optimizer0.zero_grad()
    idx = np.random.randint(len(ds))
    img_np, cam = ds[idx]
    gt = torch.from_numpy(img_np).unsqueeze(0).to(device)

    rendered, mask = renderer0.render(sh0_param, cam)
    rendered = rendered.flip(1)
    mask = mask.flip(1)

    H, W = rendered.shape[1], rendered.shape[2]
    gt_hw = gt.permute(0, 1, 2, 3)
    gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
    gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
    gt_linear = gt_resized.clamp(0, 1).pow(2.2)

    loss = criterion(rendered, gt_linear, mask, sh0_param)
    loss.backward()
    optimizer0.step()

    if step == 0 or (step + 1) % 50 == 0:
        # PSNR
        with torch.no_grad():
            _img, _cam = ds[0]
            _gt = torch.from_numpy(_img).unsqueeze(0).to(device)
            _r, _m = renderer0.render(sh0_param, _cam)
            _r = _r.flip(1); _m = _m.flip(1)
            _gt_h = F.interpolate(_gt.permute(0,1,2,3), size=(_r.shape[1], _r.shape[2]), mode="bilinear", align_corners=False)
            _gt_h = _gt_h.squeeze(0).permute(1,2,0).unsqueeze(0).pow(2.2)
            _mf = _m.unsqueeze(-1).float()
            nv = _m.sum() * 3 + 1e-8
            mse = ((_r - _gt_h) * _mf).pow(2).sum() / nv
            psnr = 10.0 * torch.log10(1.0 / mse).item() if mse > 0 else 0
        print(f"  step {step+1}: loss={loss.item():.6f} psnr={psnr:.2f}dB")

# Save SH0 render
with torch.no_grad():
    img_np0, cam0 = ds[0]
    r0, m0 = renderer0.render(sh0_param, cam0)
    r0 = r0.flip(1); m0 = m0.flip(1)
    r0_np = r0[0].clamp(0, 1).pow(1.0/2.2).cpu().numpy()
    r0_np = (r0_np * 255).astype(np.uint8)
    r0_bgr = cv2.cvtColor(r0_np, cv2.COLOR_RGB2BGR)
    m0_np = m0[0].cpu().numpy()
    r0_bgr[m0_np < 0.5] = 0

    gt0_bgr = cv2.cvtColor((img_np0.transpose(1,2,0)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h = min(r0_bgr.shape[0], gt0_bgr.shape[0])
    r0r = cv2.resize(r0_bgr, (r0_bgr.shape[1]*h//r0_bgr.shape[0], h))
    g0r = cv2.resize(gt0_bgr, (gt0_bgr.shape[1]*h//gt0_bgr.shape[0], h))
    cv2.putText(g0r, "GT", (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(r0r, "SH0", (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    canvas0 = np.concatenate([g0r, r0r], axis=1)
    cv2.imwrite("output/debug_sh0_compare.png", canvas0)
    print(f"  Saved output/debug_sh0_compare.png")

# Check SH0 texture values
dc0 = sh0_param[0, :, :, 0:3]
print(f"  SH0 DC range: [{dc0.min():.4f}, {dc0.max():.4f}] mean={dc0.mean():.4f}")
# After * C0: what's the effective diffuse?
from src.sh import _C0
diffuse = dc0 * _C0
print(f"  Diffuse (dc*C0) range: [{diffuse.min():.4f}, {diffuse.max():.4f}] mean={diffuse.mean():.4f}")
# In sRGB:
diffuse_srgb = diffuse.clamp(0,1).pow(1.0/2.2)
print(f"  Diffuse sRGB range: [{diffuse_srgb.min():.4f}, {diffuse_srgb.max():.4f}] mean={diffuse_srgb.mean():.4f}")

# Check rendered vs GT pixel stats on object
with torch.no_grad():
    valid = m0[0].cpu().numpy() > 0.5
    r_obj = r0[0].clamp(0,1).pow(1.0/2.2).cpu().numpy()[valid]
    gt_srgb = img_np0.transpose(1,2,0)
    gt_srgb_resized = cv2.resize(gt_srgb, (512, 512))
    g_obj = gt_srgb_resized[valid]
    print(f"\n  Render obj sRGB mean: [{r_obj[:,0].mean():.4f}, {r_obj[:,1].mean():.4f}, {r_obj[:,2].mean():.4f}]")
    print(f"  GT obj sRGB mean:     [{g_obj[:,0].mean():.4f}, {g_obj[:,1].mean():.4f}, {g_obj[:,2].mean():.4f}]")
    print(f"  Render obj max: {r_obj.max():.4f}")
    print(f"  GT obj max: {g_obj.max():.4f}")
