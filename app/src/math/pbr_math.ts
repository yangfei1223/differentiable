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

export interface DecodedMaterial {
  baseColor: [number, number, number]; // linear
  roughness: number;
  metallic: number;
  normalTS: [number, number, number]; // unit vector
}

export interface MaterialTextureInput {
  baseColorSRGB: [number, number, number];
  roughness: number;
  metallic: number;
  normalMap: [number, number, number]; // [0,1] range
}

/**
 * Decode 4-channel material texture samples.
 * Mirrors GLSL: shaders/pbr.frag → step 1 (Material decode)
 * Mirrors Python: src/shading/pbr/material.py → decode_material
 *
 * Note: Python's training stores raw texture as sigmoid-encoded; the exported
 * PNGs are already in display space (base_color=sRGB, roughness/metallic=linear).
 * So we only apply pow(2.2) to base_color here.
 */
export function decodeMaterial(input: MaterialTextureInput): DecodedMaterial {
  const [r, g, b] = input.baseColorSRGB;
  const baseColor: [number, number, number] = [
    Math.pow(r, 2.2),
    Math.pow(g, 2.2),
    Math.pow(b, 2.2),
  ];
  const roughness = input.roughness;
  const metallic = input.metallic;

  // Remap [0,1] → [-1,1], then normalize (mirror F.normalize)
  const nx = input.normalMap[0] * 2 - 1;
  const ny = input.normalMap[1] * 2 - 1;
  const nz = input.normalMap[2] * 2 - 1;
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  const normalTS: [number, number, number] = [nx / len, ny / len, nz / len];

  return { baseColor, roughness, metallic, normalTS };
}

/**
 * Convert [0,1] UV to align_corners=True compatible UV for WebGL textures.
 *
 * Python's grid_sample uses align_corners=True (extents map to texel centers).
 * WebGL's texture() uses align_corners=False (extents map to texel edges).
 *
 * To get equivalent sampling, transform: uv → (uv * (size-1) + 0.5) / size.
 *
 * Mirrors GLSL: shaders/pbr.frag → step 5 (BRDF LUT UV fix)
 */
export function brdfLutUVAlignCorners(
  u: number,
  v: number,
  size: number,
): [number, number] {
  const uFixed = (u * (size - 1) + 0.5) / size;
  const vFixed = (v * (size - 1) + 0.5) / size;
  return [uFixed, vFixed];
}

export interface SplitSumInput {
  baseColor: [number, number, number];
  roughness: number;
  metallic: number;
  NdotV: number;
  brdfLutScale: number; // RG.R of BRDF LUT
  brdfLutBias: number;  // RG.G of BRDF LUT
  irradiance: [number, number, number];
  prefiltered: [number, number, number];
}

/**
 * Compute final shaded RGB via split-sum composition.
 * Mirrors GLSL: shaders/pbr.frag → steps 4-6
 * Mirrors Python: src/shading/pbr_model.py:91-107 (shade_submesh)
 */
export function splitSumShade(input: SplitSumInput): [number, number, number] {
  const { baseColor, metallic, NdotV, brdfLutScale, brdfLutBias, irradiance, prefiltered } = input;

  // F0 = mix(0.04, baseColor, metallic)
  const F0: [number, number, number] = [
    0.04 * (1 - metallic) + baseColor[0] * metallic,
    0.04 * (1 - metallic) + baseColor[1] * metallic,
    0.04 * (1 - metallic) + baseColor[2] * metallic,
  ];

  // Diffuse: kd = (1 - metallic) * (1 - F0)
  const kd = (1 - metallic);
  const diffuse: [number, number, number] = [
    kd * (1 - F0[0]) * baseColor[0] * irradiance[0],
    kd * (1 - F0[1]) * baseColor[1] * irradiance[1],
    kd * (1 - F0[2]) * baseColor[2] * irradiance[2],
  ];

  // Specular: (F0 * scale + bias) * prefiltered
  const specular: [number, number, number] = [
    (F0[0] * brdfLutScale + brdfLutBias) * prefiltered[0],
    (F0[1] * brdfLutScale + brdfLutBias) * prefiltered[1],
    (F0[2] * brdfLutScale + brdfLutBias) * prefiltered[2],
  ];

  // Combine + clamp
  return [
    Math.max(0, Math.min(1, diffuse[0] + specular[0])),
    Math.max(0, Math.min(1, diffuse[1] + specular[1])),
    Math.max(0, Math.min(1, diffuse[2] + specular[2])),
  ];
}
