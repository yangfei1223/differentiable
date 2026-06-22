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
  vec3 irradiance = textureLod(uEnvMap, direction_to_uv(N), uDiffuseMipBias).rgb;
  vec3 diffuse = kd * baseColor * irradiance;

  // ===== 5. Specular =====
  float specLod = roughness * uMaxEnvMip;
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

  // ===== Final: apply linear -> sRGB encoding (mirror Python pbr_logger.py:162) =====
  // Python training saves rendered debug PNG as: rgb.clamp(0,1).pow(1/2.2) * 255
  // ShaderMaterial+GLSL3 with custom `out fragColor` does NOT auto-apply Three.js's
  // linearToOutputTexel conversion, so we mirror Python's pow(1/2.2) explicitly.
  vec3 encoded = pow(rgb, vec3(1.0 / 2.2));

  fragColor = vec4(encoded, 1.0);
}
