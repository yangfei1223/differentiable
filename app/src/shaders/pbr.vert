#include "common.glsl"

// NOTE: modelMatrix, modelViewMatrix, projectionMatrix, normalMatrix,
// cameraPosition, position, uv, normal are auto-injected by Three.js
// ShaderMaterial in GLSL3 mode (via #define attribute in).

// Outputs to fragment shader
out vec2 vUV;
out vec3 vNormalW;
out vec3 vWorldPos;
out vec3 vViewDirW;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPos = worldPos.xyz;

  // Wrap UV into [0,1] — some glTF exports emit UVs in [1,2] range (offset by 1)
  // which combined with ClampToEdge produces single-color sampling.
  vUV = fract(uv);
  vNormalW = normalize(mat3(modelMatrix) * normal);

  // View direction (camera→vertex), matching Python renderer.py:118
  vViewDirW = normalize(cameraPosition - worldPos.xyz);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
