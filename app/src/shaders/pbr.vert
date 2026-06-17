#version 300 es

precision highp float;

#include "common.glsl"

// Vertex attributes (Three.js default names)
in vec3 position;
in vec2 uv;
in vec3 normal;
in vec4 tangent; // xyz=dir, w=sign for bitangent handedness

// Uniforms (Three.js auto-injects these)
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;

// Outputs to fragment shader
out vec2 vUV;
out vec3 vNormalW;
out vec3 vTangentW;
out vec3 vBitangentW;
out vec3 vViewDirW;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);

  vUV = uv;
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
