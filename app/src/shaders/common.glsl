// Common GLSL constants and helpers — included by pbr.vert and pbr.frag.
// Portable to native (Vulkan/Metal/GLES) with minimal changes.

const float PI = 3.14159265359;

// Convert a direction vector to equirectangular UV coordinates.
// Mirrors Python EnvironmentMap.direction_to_uv (src/shading/pbr/env_map.py:60).
vec2 direction_to_uv(vec3 dir) {
  float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
  float v = asin(clamp(dir.y, -0.999, 0.999)) / PI + 0.5;
  return vec2(u, v);
}
