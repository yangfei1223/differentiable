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
uniform int   uDebug;            // 0=off, 1=baseColor, 2=roughness, 3=metallic,
                                 // 4=normalTS, 5=normalW, 6=NdotV, 7=irradiance,
                                 // 8=prefiltered, 9=diffuse, 10=specular, 11=brdf

void main() {
  vec3 N = normalize(vNormalW);
  vec3 T = normalize(vTangentW);
  vec3 B = normalize(vBitangentW);
  vec3 V = normalize(vViewDirW);

  // ===== 1. Material decode =====
  vec3 baseColor = texture(uBaseColor, vUV).rgb;
  float roughness = texture(uRoughness, vUV).r;
  float metallic = texture(uMetallic, vUV).r;
  vec3 normalTS = texture(uNormalMap, vUV).rgb * 2.0 - 1.0;
  normalTS = normalize(normalTS);

  // ===== 2. Normal mapping =====
  if (uNormalMapEnabled) {
    N = normalize(T * normalTS.x + B * normalTS.y + N * normalTS.z);
  }

  // ===== 3. Reflect direction =====
  float NdotV = clamp(dot(N, V), 0.0, 1.0);
  vec3 R = 2.0 * NdotV * N - V;
  R = normalize(R);

  // ===== 4. F0 + Diffuse =====
  vec3 F0 = mix(vec3(0.04), baseColor, metallic);
  vec3 kd = (1.0 - metallic) * (1.0 - F0);
  // Python sample_diffuse(): mip_level = max_mip (absolute, 1x1 average).
  // GLSL texture(s, uv, bias) adds bias to auto-LOD which can give a different
  // effective mip than max_mip. Use textureLod with absolute LOD = uMaxEnvMip
  // to guarantee we sample the most-blurred mip (global average of env map).
  vec3 irradiance = textureLod(uEnvMap, direction_to_uv(N), uMaxEnvMip).rgb;
  vec3 diffuse = kd * baseColor * irradiance;

  // ===== 5. Specular =====
  // Match Python env_map.sample_specular(): mip_level_bias = roughness * max_mip.
  // nvdiffrast treats this as bias (added to auto-LOD). textureLod uses absolute LOD.
  // On a curved surface, R direction varies rapidly between adjacent pixels, so
  // the auto-LOD computed by nvdiffrast from UV derivatives can be substantial.
  // We approximate Python's behavior by using GLSL texture() with bias (auto-LOD + bias).
  // texture(s, uv, bias) computes auto-LOD from screen derivatives and adds the bias.
  // Empirically: texture() alone gives ~B/G=0.73 vs GT 0.91 — bias contribution helps
  // but isn't enough. May need to amplify bias to compensate for differences in
  // nvdiffrast's auto-LOD formula vs GLSL's.
  float specLod = roughness * uMaxEnvMip;
  // DEBUG: try floor() to force integer mip (skip trilinear interpolation)
  // vec3 prefiltered = textureLod(uEnvMap, direction_to_uv(R), floor(specLod)).rgb;
  vec3 prefiltered = textureLod(uEnvMap, direction_to_uv(R), specLod).rgb;

  vec2 brdfUv = (vec2(NdotV, roughness) * (uBRDFLutSize - 1.0) + 0.5) / uBRDFLutSize;
  vec2 brdf = texture(uBRDFLut, brdfUv).rg;
  vec3 specular = (F0 * brdf.x + brdf.y) * prefiltered;

  // ===== 6. Combine =====
  vec3 rgb = diffuse + specular;
  rgb = clamp(rgb, 0.0, 1.0);

  // ===== DEBUG OUTPUT (raw values) =====
  if (uDebug == 1) { fragColor = vec4(baseColor, 1.0); return; }
  if (uDebug == 2) { fragColor = vec4(vec3(roughness), 1.0); return; }
  if (uDebug == 3) { fragColor = vec4(vec3(metallic), 1.0); return; }
  if (uDebug == 4) { fragColor = vec4(normalTS * 0.5 + 0.5, 1.0); return; }
  if (uDebug == 5) { fragColor = vec4(N * 0.5 + 0.5, 1.0); return; }
  if (uDebug == 6) { fragColor = vec4(vec3(NdotV), 1.0); return; }
  if (uDebug == 7) { fragColor = vec4(irradiance, 1.0); return; }
  if (uDebug == 8) { fragColor = vec4(prefiltered, 1.0); return; }
  if (uDebug == 9) { fragColor = vec4(diffuse, 1.0); return; }
  if (uDebug == 10) { fragColor = vec4(specular, 1.0); return; }
  if (uDebug == 11) { fragColor = vec4(brdf.x, brdf.y, 0.0, 1.0); return; }

  // ===== ENV PROBE diagnostics =====
  if (uDebug == 12) { fragColor = vec4(textureLod(uEnvMap, vec2(0.5, 0.5), 0.0).rgb, 1.0); return; }
  if (uDebug == 13) { fragColor = vec4(textureLod(uEnvMap, vec2(0.5, 0.5), uMaxEnvMip).rgb, 1.0); return; }
  if (uDebug == 14) { fragColor = vec4(textureLod(uEnvMap, vUV, 0.0).rgb, 1.0); return; }

  // ===== SHADER PROBE: show specLod and NdotV values =====
  // 15: specLod (normalized to [0,1] for visualization)
  // 16: NdotV
  // 17: roughness
  // 18: env at vUV, mip = specLod (the actual specular mip used)
  // 19: env at vUV, mip 5 (control: should be moderate blur)
  if (uDebug == 15) { fragColor = vec4(vec3(roughness * uMaxEnvMip / max(uMaxEnvMip, 1.0)), 1.0); return; }
  if (uDebug == 16) { fragColor = vec4(vec3(NdotV), 1.0); return; }
  if (uDebug == 17) { fragColor = vec4(vec3(roughness), 1.0); return; }
  if (uDebug == 18) { fragColor = vec4(textureLod(uEnvMap, direction_to_uv(R), roughness * uMaxEnvMip).rgb, 1.0); return; }
  if (uDebug == 19) { fragColor = vec4(textureLod(uEnvMap, direction_to_uv(R), 5.0).rgb, 1.0); return; }
  // 20: R direction visualization
  // 21: env at fixed uv=(0.75, 0.3) (typical "warm" region) mip 0
  // 22: env at fixed uv=(0.25, 0.7) (typical "cool" region) mip 0
  if (uDebug == 20) { fragColor = vec4(R * 0.5 + 0.5, 1.0); return; }
  if (uDebug == 21) { fragColor = vec4(textureLod(uEnvMap, vec2(0.75, 0.3), 0.0).rgb, 1.0); return; }
  if (uDebug == 22) { fragColor = vec4(textureLod(uEnvMap, vec2(0.25, 0.7), 0.0).rgb, 1.0); return; }

  // ===== PIXEL DUMP for AB comparison =====
  // Encode R direction (in [-1,1]) into 3 byte channels with bias 0.5
  // We need a way to read per-pixel R from outside the shader.
  // Mode 23: R.x encoded as red, encoded as pow(0.5+0.5*R.x, 1/2.2)*255
  // For now, just visualize prefiltered with even more controlled params
  if (uDebug == 23) {
    // Try: prefiltered with mip = pow(roughness, 2.0) * uMaxEnvMip (Epic's convention)
    float lod_alt = roughness * roughness * uMaxEnvMip;
    vec3 prefiltered_alt = textureLod(uEnvMap, direction_to_uv(R), lod_alt).rgb;
    fragColor = vec4(prefiltered_alt, 1.0);
    return;
  }
  if (uDebug == 24) {
    // Try: prefiltered with mip = (1 - NdotV) * roughness * uMaxEnvMip
    float lod_alt = (1.0 - NdotV) * roughness * uMaxEnvMip;
    vec3 prefiltered_alt = textureLod(uEnvMap, direction_to_uv(R), lod_alt).rgb;
    fragColor = vec4(prefiltered_alt, 1.0);
    return;
  }
  // 25: prefiltered at mip 5 (fixed) — control sample for fractional LOD comparison
  if (uDebug == 25) { fragColor = vec4(textureLod(uEnvMap, direction_to_uv(R), 5.0).rgb, 1.0); return; }
  // 26: prefiltered at mip 6 (fixed)
  if (uDebug == 26) { fragColor = vec4(textureLod(uEnvMap, direction_to_uv(R), 6.0).rgb, 1.0); return; }
  // 27: prefiltered at mip 7 (fixed)
  if (uDebug == 27) { fragColor = vec4(textureLod(uEnvMap, direction_to_uv(R), 7.0).rgb, 1.0); return; }
  // 28: prefiltered at mip 8 (fixed)
  if (uDebug == 28) { fragColor = vec4(textureLod(uEnvMap, direction_to_uv(R), 8.0).rgb, 1.0); return; }

  // ===== Final: apply linear -> sRGB encoding (mirror Python pbr_logger.py:162) =====
  // Python training saves rendered debug PNG as: rgb.clamp(0,1).pow(1/2.2) * 255
  // ShaderMaterial+GLSL3 with custom `out fragColor` does NOT auto-apply Three.js's
  // linearToOutputTexel conversion, so we mirror Python's pow(1/2.2) explicitly.
  vec3 encoded = pow(rgb, vec3(1.0 / 2.2));

  fragColor = vec4(encoded, 1.0);
}
