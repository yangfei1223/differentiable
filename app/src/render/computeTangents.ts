/**
 * Compute vertex tangents for a THREE.BufferGeometry.
 *
 * Mirrors Python src/mesh.py:78-141 (compute_vertex_tangents) exactly:
 *   - For each triangle: face_tangent = (dv2*e1 - dv1*e2) / det
 *   - Area-weighted accumulation to vertices
 *   - Gram-Schmidt orthogonalize T against N
 *   - bitangent = cross(N, T)  (NO tangent.w multiplication)
 *
 * Output: sets `tangent` attribute as Float32Array [nx, ny, nz, w=1].
 * The w component is always 1 (Python doesn't use handedness), but the
 * attribute is vec4 for compatibility with standard glTF conventions.
 *
 * IMPORTANT: bitangent in shader should be computed as cross(N, T.xyz)
 * WITHOUT multiplying by tangent.w, to match Python.
 */
import * as THREE from 'three';

export function computeVertexTangents(geometry: THREE.BufferGeometry): void {
  if (!geometry.hasAttribute('position') || !geometry.hasAttribute('normal') || !geometry.hasAttribute('uv')) {
    throw new Error('computeVertexTangents requires position, normal, uv attributes');
  }

  const position = geometry.attributes.position;
  const normal = geometry.attributes.normal;
  const uv = geometry.attributes.uv;
  const index = geometry.index;

  const vertexCount = position.count;
  const tangents = new Float32Array(vertexCount * 4); // xyzw per vertex
  const bitangents = new Float32Array(vertexCount * 3);

  // Helper: get position/normal/uv by vertex index
  const getPos = (i: number, out: THREE.Vector3) =>
    out.set(position.getX(i), position.getY(i), position.getZ(i));
  const getNorm = (i: number, out: THREE.Vector3) =>
    out.set(normal.getX(i), normal.getY(i), normal.getZ(i));
  const getUV = (i: number, out: THREE.Vector2) =>
    out.set(uv.getX(i), uv.getY(i));

  // Iterate triangles
  const triCount = index ? index.count / 3 : vertexCount / 3;
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const faceTangent = new THREE.Vector3();
  const faceBitangent = new THREE.Vector3();
  const uv0 = new THREE.Vector2();
  const uv1 = new THREE.Vector2();
  const uv2 = new THREE.Vector2();
  const cross = new THREE.Vector3();

  for (let t = 0; t < triCount; t++) {
    let i0: number, i1: number, i2: number;
    if (index) {
      i0 = index.getX(t * 3);
      i1 = index.getX(t * 3 + 1);
      i2 = index.getX(t * 3 + 2);
    } else {
      i0 = t * 3;
      i1 = t * 3 + 1;
      i2 = t * 3 + 2;
    }

    getPos(i0, v0);
    getPos(i1, v1);
    getPos(i2, v2);
    e1.subVectors(v1, v0);
    e2.subVectors(v2, v0);

    getUV(i0, uv0);
    getUV(i1, uv1);
    getUV(i2, uv2);
    const du1 = uv1.x - uv0.x;
    const dv1 = uv1.y - uv0.y;
    const du2 = uv2.x - uv0.x;
    const dv2 = uv2.y - uv0.y;

    const det = du1 * dv2 - du2 * dv1;
    if (Math.abs(det) < 1e-10) continue;
    const invdet = 1.0 / det;

    // face_tangent = (dv2 * e1 - dv1 * e2) * invdet  (Python mesh.py:112)
    faceTangent.copy(e1).multiplyScalar(dv2).addScaledVector(e2, -dv1).multiplyScalar(invdet);
    // face_bitangent = (-du2 * e1 + du1 * e2) * invdet
    faceBitangent.copy(e1).multiplyScalar(-du2).addScaledVector(e2, du1).multiplyScalar(invdet);

    // Area = 0.5 * |cross(e1, e2)|
    cross.crossVectors(e1, e2);
    const area = cross.length() * 0.5;

    // Accumulate area-weighted to each vertex of the triangle
    for (const vi of [i0, i1, i2]) {
      tangents[vi * 4] += faceTangent.x * area;
      tangents[vi * 4 + 1] += faceTangent.y * area;
      tangents[vi * 4 + 2] += faceTangent.z * area;
      bitangents[vi * 3] += faceBitangent.x * area;
      bitangents[vi * 3 + 1] += faceBitangent.y * area;
      bitangents[vi * 3 + 2] += faceBitangent.z * area;
    }
  }

  // Gram-Schmidt orthogonalize + normalize
  const n = new THREE.Vector3();
  const t = new THREE.Vector3();
  for (let i = 0; i < vertexCount; i++) {
    getNorm(i, n);
    t.set(tangents[i * 4], tangents[i * 4 + 1], tangents[i * 4 + 2]);

    // t = t - dot(t, n) * n  (Python mesh.py:127)
    t.addScaledVector(n, -t.dot(n));
    const tn = t.length();
    if (tn > 1e-10) {
      t.divideScalar(tn);
    } else {
      // Degenerate: pick arbitrary perpendicular
      if (Math.abs(n.x) < 0.9) {
        t.set(1, 0, 0);
      } else {
        t.set(0, 1, 0);
      }
      t.addScaledVector(n, -t.dot(n)).normalize();
    }

    tangents[i * 4] = t.x;
    tangents[i * 4 + 1] = t.y;
    tangents[i * 4 + 2] = t.z;
    tangents[i * 4 + 3] = 1.0; // w = 1 (Python doesn't use handedness)
  }

  geometry.setAttribute('tangent', new THREE.BufferAttribute(tangents, 4));
}
