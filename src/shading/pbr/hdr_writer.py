"""Write HDR images in Radiance RGBE format.

Radiance .hdr format:
  Header (text):
    #?RADIANCE\n
    GAMMA=1.0\n
    FORMAT=32-bit_rle_rgbe\n
    \n
    -Y H +X W\n
  Body: per-scanline RLE-encoded RGBE bytes

Three.js RGBELoader reads this natively.
"""
from __future__ import annotations

import numpy as np


def _float_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    """Convert [W, 3] float to [W, 4] uint8 RGBE.

    Radiance RGBE encoding:
        shared exponent e = ceil(log2(max(R,G,B)) + 128), clamped to [1, 255]
        mantissa = round(channel * 256 / 2^(e-128))
        decode:  channel = mantissa * 2^(e-128) / 256
    Black pixels (max <= 1e-32) become (0,0,0,0).
    """
    W = rgb.shape[0]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    nonzero = maxc > 1e-32

    e = np.zeros(W, dtype=np.int32)
    e[nonzero] = np.ceil(np.log2(maxc[nonzero]) + 128).astype(np.int32)
    e = np.clip(e, 1, 255)  # exponent must be >= 1 for non-black

    # mantissa = round(channel * 256 / 2^(e-128))
    # Compute 2^(e-128) safely. e in [1, 255] so exponent in [-127, 127],
    # but float32 underflow at very low values can still produce NaN when
    # multiplied. Use float64 for the scale calculation, then cast result.
    pow_scale = np.power(2.0, (e - 128).astype(np.float64))  # float64 to avoid overflow
    inv_scale = 256.0 / pow_scale
    # Replace inf/nan from divide-by-zero (black pixels handled below)
    inv_scale = np.where(np.isfinite(inv_scale), inv_scale, 0.0)

    rgbe = np.empty((W, 4), dtype=np.uint8)
    rgbe[:, 0] = np.clip(np.round(r.astype(np.float64) * inv_scale), 0, 255).astype(np.uint8)
    rgbe[:, 1] = np.clip(np.round(g.astype(np.float64) * inv_scale), 0, 255).astype(np.uint8)
    rgbe[:, 2] = np.clip(np.round(b.astype(np.float64) * inv_scale), 0, 255).astype(np.uint8)
    rgbe[:, 3] = e.astype(np.uint8)
    rgbe[~nonzero] = 0
    return rgbe


def _rle_encode_scanline(rgbe: np.ndarray) -> bytes:
    """RLE-encode one RGBE scanline (Greg Ward's new-style RLE).

    For W <= 8 or W > 32767, falls back to raw (unencoded) — Three.js handles both.
    Otherwise emits the 4-byte header [2, 2, W_hi, W_lo] then per-channel RLE.
    """
    W = rgbe.shape[0]
    if W < 8 or W > 32767:
        return rgbe.tobytes()

    out = bytearray()
    out.append(2)
    out.append(2)
    out.append((W >> 8) & 0xFF)
    out.append(W & 0xFF)

    for c in range(4):
        ch = rgbe[:, c]
        cur = 0
        while cur < W:
            # Detect a run of identical values (length 1..127)
            run_start = cur
            while cur < W and (cur - run_start) < 127 and ch[cur] == ch[run_start]:
                cur += 1
            run_len = cur - run_start

            if run_len >= 4:
                # RLE: high bit set + count, followed by value
                out.append(128 | run_len)
                out.append(int(ch[run_start]))
            else:
                # Literal: gather until next 4+ run begins, up to 127
                lit_start = run_start
                # cur already advanced by run_len; keep extending while no run
                while cur < W and (cur - lit_start) < 127:
                    # Look ahead: would a 4+ run start here?
                    if (cur + 3 < W
                            and ch[cur] == ch[cur + 1]
                            and ch[cur] == ch[cur + 2]
                            and ch[cur] == ch[cur + 3]):
                        break
                    cur += 1
                lit_len = cur - lit_start
                out.append(lit_len)
                out.extend(ch[lit_start:lit_start + lit_len].tobytes())
    return bytes(out)


def write_rgbe_hdr(path: str, image: np.ndarray) -> None:
    """Write an HDR image to a Radiance .hdr file.

    Args:
        path: Output file path.
        image: HDR image [H, W, 3], float32, non-negative (values > 1 OK).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] image, got shape {image.shape}")
    H, W = image.shape[:2]
    img = np.ascontiguousarray(image, dtype=np.float32)

    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"GAMMA=1.0\n")
        f.write(b"EXPOSURE=1.0\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        f.write(b"\n")
        f.write(f"-Y {H} +X {W}\n".encode("ascii"))
        for y in range(H):
            rgbe = _float_to_rgbe(img[y])
            f.write(_rle_encode_scanline(rgbe))


def write_hdr_from_tensor(path: str, decoded_tensor) -> None:
    """Write a torch tensor [1, H, W, 3] (or [H, W, 3]) of softplus-decoded HDR to .hdr."""
    import torch
    if isinstance(decoded_tensor, torch.Tensor):
        arr = decoded_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(decoded_tensor)
    if arr.ndim == 4:
        arr = arr[0]
    write_rgbe_hdr(path, arr)
