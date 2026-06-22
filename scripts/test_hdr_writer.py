"""Smoke test the RGBE writer.

Tests:
1. Round-trip a simple HDR image through writer + a minimal RGBE reader.
2. Verify Three.js-compatible header.
3. Test with the actual env_map data (max=17.27).
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import torch.nn.functional as F
from src.shading.pbr.hdr_writer import write_rgbe_hdr, _float_to_rgbe


def simple_rgbe_decode(data: bytes) -> np.ndarray:
    """Minimal RGBE reader for round-trip verification."""
    # Parse header
    pos = 0
    nl = data.index(b'\n', pos) + 1
    assert data[:nl] == b'#?RADIANCE\n', f"bad magic: {data[:nl]}"
    pos = nl
    # Skip header lines until empty line
    while True:
        nl = data.index(b'\n', pos) + 1
        line = data[pos:nl-1]
        pos = nl
        if line == b'':
            break
    # Resolution line
    nl = data.index(b'\n', pos) + 1
    res = data[pos:nl-1].decode('ascii')
    pos = nl
    # Parse: -Y H +X W
    parts = res.replace('-', ' ').replace('+', ' ').split()
    H, W = int(parts[1]), int(parts[3])

    out = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(H):
        if data[pos] == 2 and data[pos+1] == 2 and (data[pos+2] << 8 | data[pos+3]) == W:
            # new-style RLE
            pos += 4
            for c in range(4):
                cx = 0
                while cx < W:
                    code = data[pos]; pos += 1
                    if code & 128:
                        run = code & 127
                        val = data[pos]; pos += 1
                        out[y, cx:cx+run, c] = val
                        cx += run
                    else:
                        lit = code
                        out[y, cx:cx+lit, c] = np.frombuffer(data[pos:pos+lit], dtype=np.uint8)
                        pos += lit
                        cx += lit
        else:
            # raw
            out[y] = np.frombuffer(data[pos:pos + W*4], dtype=np.uint8).reshape(W, 4)
            pos += W * 4

    # Decode RGBE → float
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    e = out[..., 3].astype(np.int32)
    scale = np.power(2.0, (e - 128).astype(np.float32)) / 256.0
    rgb[..., 0] = out[..., 0].astype(np.float32) * scale
    rgb[..., 1] = out[..., 1].astype(np.float32) * scale
    rgb[..., 2] = out[..., 2].astype(np.float32) * scale
    rgb[e == 0] = 0
    return rgb


# Test 1: simple round-trip
print("=== Test 1: round-trip simple image ===")
test = np.array([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [10.0, 5.0, 0.1]]], dtype=np.float32)
write_rgbe_hdr('/tmp/test.hdr', test)
with open('/tmp/test.hdr', 'rb') as f:
    data = f.read()
decoded = simple_rgbe_decode(data)
print(f"Original: {test[0]}")
print(f"Decoded:  {decoded[0]}")
err = np.abs(test - decoded).max()
print(f"Max abs error: {err:.4f}")
assert err < 0.05, f"Round-trip error too high: {err}"

# Test 2: actual env_map from checkpoint
print("\n=== Test 2: real env_map ===")
state = torch.load('output/helmet_260604_pbr/epoch2000/pbr_checkpoint.pt', map_location='cpu', weights_only=False)
raw = state['env_map']
decoded_full = F.softplus(raw)
print(f"Decoded shape: {decoded_full.shape}, max={decoded_full.max():.4f}, mean={decoded_full.mean():.4f}")

# Write and read back
write_rgbe_hdr('/tmp/env_map.hdr', decoded_full[0].numpy())
with open('/tmp/env_map.hdr', 'rb') as f:
    data = f.read()
print(f"HDR file size: {len(data)} bytes")
decoded_back = simple_rgbe_decode(data)
print(f"Decoded back shape: {decoded_back.shape}, max={decoded_back.max():.4f}, mean={decoded_back.mean():.4f}")
err = (decoded_full[0].numpy() - decoded_back).max()
print(f"Max abs error: {err:.4f}")
rel_err = ((decoded_full[0].numpy() - decoded_back) / (decoded_full[0].numpy() + 1e-6)).mean()
print(f"Mean rel error: {rel_err:.6f}")

# Per-channel comparison
print("\nPer-channel mean comparison (original vs decoded):")
print(f"  R: {decoded_full[0].mean(dim=(0,1))[0]:.4f} vs {decoded_back[..., 0].mean():.4f}")
print(f"  G: {decoded_full[0].mean(dim=(0,1))[1]:.4f} vs {decoded_back[..., 1].mean():.4f}")
print(f"  B: {decoded_full[0].mean(dim=(0,1))[2]:.4f} vs {decoded_back[..., 2].mean():.4f}")
