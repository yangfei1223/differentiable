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

