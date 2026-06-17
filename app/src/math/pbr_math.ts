/**
 * TypeScript mirror of PBR GLSL math.
 *
 * These functions are NOT used at runtime (the GLSL shaders are).
 * They exist solely for unit testing — to verify the GLSL logic
 * produces values matching Python's src/shading/pbr_model.py.
 *
 * Each function documents the GLSL line it mirrors.
 */

export const PI = Math.PI;

/**
 * Convert direction vector to equirectangular UV.
 * Mirrors GLSL: shaders/common.glsl → direction_to_uv
 * Mirrors Python: src/shading/pbr/env_map.py:60 → EnvironmentMap.direction_to_uv
 */
export function directionToUV(dir: [number, number, number]): [number, number] {
  const [x, y, z] = dir;
  const u = Math.atan2(z, x) / (2 * PI) + 0.5;
  const yClamped = Math.max(-0.999, Math.min(0.999, y));
  const v = Math.asin(yClamped) / PI + 0.5;
  return [u, v];
}
