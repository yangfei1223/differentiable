#version 300 es

precision highp float;

#include "common.glsl"

// Inputs from vertex shader
in vec2 vUV;
in vec3 vNormalW;
in vec3 vTangentW;
in vec3 vBitangentW;
in vec3 vViewDirW;

// Output
out vec4 fragColor;

// Material textures
uniform sampler2D uBaseColor;    // sRGB
uniform sampler2D uRoughness;    // R channel, linear
uniform sampler2D uMetallic;     // R channel, linear
uniform sampler2D uNormalMap;    // [0,1] encoding of [-1,1] tangent-space normal

// Environment
uniform sampler2D uEnvMap;       // equirect, RGBA, will be sampled with mipmaps
uniform sampler2D uBRDFLut;      // 2-channel (RG) lookup table

// Runtime parameters
uniform float uMaxEnvMip;        // = floor(log2(max(envH, envW)))
uniform float uDiffuseMipBias;   // = uMaxEnvMip (sample most-blurred mip)
uniform bool  uNormalMapEnabled;
uniform vec2  uBRDFLutSize;      // e.g. vec2(256.0, 256.0)

void main() {
  vec3 N = normalize(vNormalW);
  vec3 T = normalize(vTangentW);
  vec3 B = normalize(vBitangentW);
  vec3 V = normalize(vViewDirW);

  // ===== 1. Material decode (mirror Python: src/shading/pbr/material.py -> decode_material) =====
  vec3 baseColor = pow(texture(uBaseColor, vUV).rgb, vec3(2.2));
  float roughness = texture(uRoughness, vUV).r;
  float metallic = texture(uMetallic, vUV).r;
  vec3 normalTS = texture(uNormalMap, vUV).rgb * 2.0 - 1.0;
  normalTS = normalize(normalTS); // mirror F.normalize

  // ===== 2. Normal mapping (direct TBN, no Gram-Schmidt) =====
  // Mirror Python: src/shading/pbr_model.py:79-86 (shade_submesh)
  if (uNormalMapEnabled) {
    N = normalize(T * normalTS.x + B * normalTS.y + N * normalTS.z);
  }

  // ===== 3. Reflect direction (mirror shade_submesh lines 89-92) =====
  float NdotV = clamp(dot(N, V), 0.0, 1.0);
  vec3 R = 2.0 * NdotV * N - V;
  R = normalize(R);

  // ===== 4. F0 + Diffuse (mirror compute_F0 + shade_submesh lines 95-99) =====
  vec3 F0 = mix(vec3(0.04), baseColor, metallic); // dielectric_F0 = 0.04
  vec3 kd = (1.0 - metallic) * (1.0 - F0);
  vec3 irradiance = textureLod(uEnvMap, direction_to_uv(N), uDiffuseMipBias).rgb;
  vec3 diffuse = kd * baseColor * irradiance;

  // ===== 5. Specular (mirror shade_submesh lines 102-108) =====
  float specLod = roughness * uMaxEnvMip; // linear mapping (NOT roughness^2)
  vec3 prefiltered = textureLod(uEnvMap, direction_to_uv(R), specLod).rgb;

  // BRDF LUT sampling with align_corners fix:
  // Python uses grid_sample(align_corners=True); WebGL texture() is align_corners=False.
  // Transform: uv -> (uv * (size-1) + 0.5) / size
  vec2 brdfUv = (vec2(NdotV, roughness) * (uBRDFLutSize - 1.0) + 0.5) / uBRDFLutSize;
  vec2 brdf = texture(uBRDFLut, brdfUv).rg;
  vec3 specular = (F0 * brdf.x + brdf.y) * prefiltered;

  // ===== 6. Combine (mirror shade_submesh lines 111-113) =====
  vec3 rgb = diffuse + specular;
  rgb = clamp(rgb, 0.0, 1.0);

  fragColor = vec4(rgb, 1.0);
}
