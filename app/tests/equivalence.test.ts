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
