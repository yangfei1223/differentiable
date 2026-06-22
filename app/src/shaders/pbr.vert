#include "common.glsl"

// Custom vertex attribute (geometry has tangent, but ShaderMaterial doesn't
// trigger Three.js USE_TANGENT define since we use custom uNormalMap uniform,
// not material.normalMap). We declare it manually.
in vec4 tangent;

// position/uv/normal/modelMatrix/modelViewMatrix/projectionMatrix/cameraPosition
// are auto-injected by Three.js ShaderMaterial.

// Outputs to fragment shader
out vec2 vUV;
out vec3 vNormalW;
out vec3 vTangentW;
out vec3 vBitangentW;
out vec3 vViewDirW;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);

  // Wrap UV into [0,1] — some glTF exports emit UVs in [1,2] range (offset by 1)
  vUV = fract(uv);
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vTangentW = normalize(mat3(modelMatrix) * tangent.xyz);

  // Bitangent: cross(N, T) — matches Python src/mesh.py:139 (no tangent.w multiplication)
  vBitangentW = normalize(cross(vNormalW, vTangentW));

  // View direction: camera→vertex, matches Python renderer.py:118
  vViewDirW = normalize(cameraPosition - worldPos.xyz);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
