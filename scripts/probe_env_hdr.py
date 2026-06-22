"""Check the actual HDR range of the trained env_map raw values.

Critical question: does softplus(raw) produce values >1.0?
If yes, the PNG export (clamped to [0,1]) is LOSSY, and the Python
shading at training time used the unclamped HDR values, which would
explain why Python render is neutral gray but Web render is warm-brown
(warm-biased values get clamped, distorting the color balance).
"""
import torch
import sys, os
sys.path.insert(0, os.path.abspath('.'))

# Find pbr checkpoint
ckpt_path = 'output/helmet_260604_pbr/epoch2000/pbr_checkpoint.pt'
if not os.path.exists(ckpt_path):
    import glob
    ckpts = sorted(glob.glob('output/helmet_260604_pbr/epoch*/pbr_checkpoint.pt'))
    print('Available checkpoints:', ckpts[-5:])
    if ckpts:
        ckpt_path = ckpts[-1]

print(f'Loading: {ckpt_path}')
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('State keys:', list(state.keys()))

if 'env_map' in state:
    raw = state['env_map']
    print(f'env_map raw shape: {raw.shape}, dtype: {raw.dtype}')
    import torch.nn.functional as F
    decoded = F.softplus(raw)
    print(f'decoded (softplus):')
    print(f'  min:  {decoded.min().item():.4f}')
    print(f'  max:  {decoded.max().item():.4f}')
    print(f'  mean: {decoded.mean().item():.4f}')
    print(f'  per-channel mean: {decoded.mean(dim=(0,1,2)).tolist()}')
    print(f'  % > 1.0: {(decoded > 1.0).float().mean().item()*100:.2f}%')
    print(f'  % > 0.9: {(decoded > 0.9).float().mean().item()*100:.2f}%')
    print(f'  clamp(0,1) loss (L2 sum): {((decoded.clamp(0,1) - decoded) ** 2).sum().item():.4f}')
    
    # Per-channel stats after clamp (what PNG stores)
    clamped = decoded.clamp(0, 1)
    print(f'\n  After clamp(0,1) per-channel mean: {clamped.mean(dim=(0,1,2)).tolist()}')
    print(f'  RAW per-channel mean:                {decoded.mean(dim=(0,1,2)).tolist()}')
    print(f'  Ratio raw/clamped:                   {(decoded.mean(dim=(0,1,2)) / clamped.mean(dim=(0,1,2))).tolist()}')
