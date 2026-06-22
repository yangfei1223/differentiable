#include "common.glsl"

// Custom attribute (Three.js auto-injects position/uv/normal in ShaderMaterial).
in vec4 tangent; // xyz=dir, w=sign for bitangent handedness

// NOTE: modelMatrix, modelViewMatrix, projectionMatrix, normalMatrix,
// cameraPosition are auto-injected by Three.js for ShaderMaterial.

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
  // fract() is the GLSL idiom for mod(x, 1.0).
  vUV = fract(uv);
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vTangentW = normalize(mat3(modelMatrix) * tangent.xyz);

  // Bitangent: cross(N, T) * tangent.w (glTF convention)
  // Mirrors Python: src/mesh.py -> compute_vertex_tangents -> B = cross(N, T)
  vBitangentW = normalize(cross(vNormalW, vTangentW) * tangent.w);

  // View direction: from camera to fragment, in world space
  // Mirrors Python: view_dirs is normalized (camera-to-vertex)
  vViewDirW = normalize(cameraPosition - worldPos.xyz);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
