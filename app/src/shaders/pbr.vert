#include "common.glsl"

// NOTE: modelMatrix, modelViewMatrix, projectionMatrix, normalMatrix,
// cameraPosition, position, uv, normal are auto-injected by Three.js
// ShaderMaterial in GLSL3 mode (via #define attribute in).

// Outputs to fragment shader
out vec2 vUV;
out vec3 vNormalW;
out vec3 vTangentW;
out vec3 vBitangentW;
out vec3 vViewDirW;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);

  // Wrap UV into [0,1] — some glTF exports emit UVs in [1,2] range (offset by 1)
  // which combined with ClampToEdge produces single-color sampling.
  vUV = fract(uv);
  vNormalW = normalize(mat3(modelMatrix) * normal);

  // Build tangent space from world-space normal (Mikktspace fallback).
  // Source mesh has no TANGENT attribute, so we synthesize an orthonormal
  // TBN basis: pick a vector not parallel to N, project to perpendicular,
  // normalize → T. Bitangent = cross(N, T) (matches Python src/mesh.py:139,
  // which also does NOT multiply by tangent.w).
  vec3 absN = abs(vNormalW);
  vec3 ref = (absN.x < 0.9) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  vTangentW = normalize(ref - dot(ref, vNormalW) * vNormalW);
  vBitangentW = normalize(cross(vNormalW, vTangentW));

  // View direction: from camera to fragment, in world space
  // Mirrors Python: view_dirs is normalized (camera-to-vertex)
  vViewDirW = normalize(cameraPosition - worldPos.xyz);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
