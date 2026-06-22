"""Render Web viewer at camera[50] (== compare_0001.png) and pixel-compare.

Python training pipeline renders the helmet at camera[50] and saves
compare_0001.png with 4 panels (GT | Rendered | Diffuse | Specular).
We crop the 'Rendered' panel (top-right of the 2x2 grid) as GT.
Then the Web viewer renders the same camera, we screenshot, and compare.
"""
import json
import numpy as np
from PIL import Image

# Load camera[50]
with open('data/helmet_260604/cameras.json') as f:
    cams = json.load(f)['cameras']
cam = cams[50]  # compare_0001

# Build URL hash (Blender Z-up coords, comma-separated)
p = cam['position']
t = cam['look_at']
u = cam['up']
fov = cam['fov_deg']
url_hash = f"#cam={p[0]},{p[1]},{p[2]},{t[0]},{t[1]},{t[2]},{u[0]},{u[1]},{u[2]},{fov}"
print(f'URL: http://localhost:5173/{url_hash}')

# Load GT (Rendered panel = top-right of compare_0001.png 2x2 grid)
gt_path = 'output/helmet_260604_pbr/epoch2000/compare_0001.png'
import os
if not os.path.exists(gt_path):
    # Fall back to epoch1000 if epoch2000 not available
    gt_path = 'output/helmet_260604_pbr/epoch1000/compare_0001.png'
print(f'GT path: {gt_path}')
gt_full = np.array(Image.open(gt_path).convert('RGB'))
h, w = gt_full.shape[:2]
# Top-right quadrant = 'Rendered' panel
gt_rendered = gt_full[:h//2, w//2:]
print(f'GT rendered panel: shape={gt_rendered.shape}')
print(f'  mean RGB: {gt_rendered.mean(axis=(0,1)).astype(int).tolist()}')
print(f'  fg mask (>30): {((gt_rendered.max(axis=2) > 30).mean() * 100):.1f}%')

# Save GT panel for later comparison
Image.fromarray(gt_rendered).save('app/debug/gt_rendered_panel.png')
print(f'Saved: app/debug/gt_rendered_panel.png')
