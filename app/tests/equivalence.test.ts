import { describe, it, expect } from 'vitest';
import { directionToUV } from '../src/math/pbr_math';

describe('direction_to_uv', () => {
  it('maps (0,1,0) to near top center', () => {
    // atan2(0, 0) = 0 → u = 0 + 0.5 = 0.5
    // y is clamped to 0.999 to avoid asin singularity
    // asin(0.999) ≈ 1.52607 → v ≈ 0.9858
    const [u, v] = directionToUV([0, 1, 0]);
    expect(u).toBeCloseTo(0.5, 5);
    expect(v).toBeCloseTo(0.9857635625937604, 5);
  });

  it('maps (1,0,0) to equator center', () => {
    // atan2(0, 1) = 0 → u = 0.5
    // asin(0) = 0 → v = 0.5
    const [u, v] = directionToUV([1, 0, 0]);
    expect(u).toBeCloseTo(0.5, 5);
    expect(v).toBeCloseTo(0.5, 5);
  });

  it('maps (0,0,1) to equator right', () => {
    // atan2(1, 0) = π/2 → u = 0.25 + 0.5 = 0.75
    // asin(0) = 0 → v = 0.5
    const [u, v] = directionToUV([0, 0, 1]);
    expect(u).toBeCloseTo(0.75, 5);
    expect(v).toBeCloseTo(0.5, 5);
  });

  it('maps (0,-1,0) to near bottom center', () => {
    // y clamped to -0.999
    const [u, v] = directionToUV([0, -1, 0]);
    expect(v).toBeCloseTo(0.014236437406239644, 5);
  });

  it('clamps extreme y values', () => {
    const [u, v] = directionToUV([0, 1.5, 0]);
    expect(v).toBeCloseTo(0.9857635625937604, 5);
  });
});
import { decodeMaterial } from '../src/math/pbr_math';

describe('decode_material', () => {
  it('decodes base_color with sRGB→linear (gamma 2.2)', () => {
    // sRGB 0.5 → linear ≈ 0.214
    const texInput = {
      baseColorSRGB: [0.5, 0.5, 0.5] as [number, number, number],
      roughness: 0.0,
      metallic: 0.0,
      normalMap: [0.5, 0.5, 1.0] as [number, number, number], // (0,0,1) after remap
    };
    const m = decodeMaterial(texInput);
    expect(m.baseColor[0]).toBeCloseTo(Math.pow(0.5, 2.2), 5);
    expect(m.baseColor[1]).toBeCloseTo(Math.pow(0.5, 2.2), 5);
  });

  it('passes roughness and metallic through directly', () => {
    const m = decodeMaterial({
      baseColorSRGB: [1, 1, 1],
      roughness: 0.7,
      metallic: 0.3,
      normalMap: [0.5, 0.5, 1.0],
    });
    expect(m.roughness).toBeCloseTo(0.7, 5);
    expect(m.metallic).toBeCloseTo(0.3, 5);
  });

  it('remaps normal from [0,1] to [-1,1] and normalizes', () => {
    const m = decodeMaterial({
      baseColorSRGB: [1, 1, 1],
      roughness: 0.5,
      metallic: 0.0,
      normalMap: [1.0, 1.0, 1.0], // → (1,1,1) / sqrt(3)
    });
    const inv = 1 / Math.sqrt(3);
    expect(m.normalTS[0]).toBeCloseTo(inv, 4);
    expect(m.normalTS[1]).toBeCloseTo(inv, 4);
    expect(m.normalTS[2]).toBeCloseTo(inv, 4);
  });

  it('neutral normal (0,0,1) stays (0,0,1)', () => {
    const m = decodeMaterial({
      baseColorSRGB: [1, 1, 1],
      roughness: 0.5,
      metallic: 0.0,
      normalMap: [0.5, 0.5, 1.0],
    });
    expect(m.normalTS[0]).toBeCloseTo(0, 5);
    expect(m.normalTS[1]).toBeCloseTo(0, 5);
    expect(m.normalTS[2]).toBeCloseTo(1, 5);
  });
});
import { brdfLutUVAlignCorners } from '../src/math/pbr_math';

describe('brdf_lut align_corners fix', () => {
  // Python uses grid_sample(align_corners=True).
  // WebGL texture() uses align_corners=False.
  // Fix: uv_fixed = (uv * (size - 1) + 0.5) / size

  it('at uv=(0,0) with size=256 maps to center of first texel', () => {
    const [u, v] = brdfLutUVAlignCorners(0, 0, 256);
    expect(u).toBeCloseTo(0.5 / 256, 5);
    expect(v).toBeCloseTo(0.5 / 256, 5);
  });

  it('at uv=(1,1) with size=256 maps to center of last texel', () => {
    const [u, v] = brdfLutUVAlignCorners(1, 1, 256);
    expect(u).toBeCloseTo(255.5 / 256, 5);
    expect(v).toBeCloseTo(255.5 / 256, 5);
  });

  it('at uv=(0.5,0.5) with size=256 maps to middle of texture', () => {
    const [u, v] = brdfLutUVAlignCorners(0.5, 0.5, 256);
    expect(u).toBeCloseTo(128.0 / 256, 5);
    expect(v).toBeCloseTo(128.0 / 256, 5);
  });

  it('preserves input range [0,1]', () => {
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const [u, v] = brdfLutUVAlignCorners(t, t, 256);
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThanOrEqual(1);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
import { splitSumShade } from '../src/math/pbr_math';

describe('split_sum composition', () => {
  const F0_dielectric = 0.04;

  it('pure diffuse (metallic=0, roughness=1) — specular ≈ 0', () => {
    const result = splitSumShade({
      baseColor: [0.8, 0.8, 0.8],
      roughness: 1.0,
      metallic: 0.0,
      NdotV: 0.9,
      brdfLutScale: 0.0, // at high roughness, scale (F0 multiplier) is small
      brdfLutBias: 0.0,
      irradiance: [1.0, 1.0, 1.0],
      prefiltered: [0.0, 0.0, 0.0], // high roughness → very blurred env
    });
    // diffuse = (1-0)*(1-0.04) * 0.8 * 1.0 = 0.768
    expect(result[0]).toBeCloseTo(0.768, 3);
    expect(result[1]).toBeCloseTo(0.768, 3);
    expect(result[2]).toBeCloseTo(0.768, 3);
  });

  it('pure metal (metallic=1) — diffuse ≈ 0, specular dominates', () => {
    const result = splitSumShade({
      baseColor: [0.95, 0.6, 0.3], // gold-ish
      roughness: 0.1,
      metallic: 1.0,
      NdotV: 0.9,
      brdfLutScale: 0.8,
      brdfLutBias: 0.05,
      irradiance: [1.0, 1.0, 1.0],
      prefiltered: [0.9, 0.9, 0.9],
    });
    // F0 = mix(0.04, baseColor, 1.0) = baseColor
    // kd = (1-1)*(1-F0) = 0 → diffuse = 0
    // specular = (F0 * 0.8 + 0.05) * prefiltered = (0.76 + 0.05 + ...) * 0.9
    expect(result[0]).toBeGreaterThan(result[1]); // red dominant
    expect(result[1]).toBeGreaterThan(result[2]);
  });

  it('clamps output to [0,1]', () => {
    const result = splitSumShade({
      baseColor: [2.0, 2.0, 2.0],
      roughness: 0.0,
      metallic: 1.0,
      NdotV: 1.0,
      brdfLutScale: 1.0,
      brdfLutBias: 1.0,
      irradiance: [10, 10, 10],
      prefiltered: [10, 10, 10],
    });
    expect(result[0]).toBeLessThanOrEqual(1.0);
    expect(result[1]).toBeLessThanOrEqual(1.0);
    expect(result[2]).toBeLessThanOrEqual(1.0);
  });
});

