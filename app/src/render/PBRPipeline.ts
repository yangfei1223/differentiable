import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import type { AssetBundle } from '../app/SceneLoader';
import { Environment } from './Environment';
import { PBRMesh } from './PBRMesh';
import { computeVertexTangents } from './computeTangents';

/**
 * Owns the Three.js renderer, scene, camera, and the currently-loaded scene's PBRMeshes.
 */
export class PBRPipeline {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  private readonly gltfLoader: GLTFLoader;
  private pbrMeshes: PBRMesh[] = [];
  private currentEnv: Environment | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);

    this.gltfLoader = new GLTFLoader();
  }

  setSize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Load and instantiate a scene from an AssetBundle.
   * Disposes any previously-loaded scene first.
   */
  async loadScene(bundle: AssetBundle): Promise<{ meshes: PBRMesh[]; env: Environment }> {
    this.disposeScene();

    // 1. Load env map + BRDF LUT
    this.currentEnv = await Environment.fromUrls(
      bundle.envMapUrl,
      bundle.brdfLutUrl,
      bundle.manifest.environment.is_hdr,
    );

    // 2. Detect BRDF LUT size from the loaded image
    const brdfLutSize = this.currentEnv.brdfLutSize;

    // 3. Load GLB
    const gltf = await this.gltfLoader.loadAsync(bundle.glbUrl);

    // 3a. Extract glTF JSON to build the REAL mesh.name → mesh index map.
    // CRITICAL: Three.js GLTFLoader assigns obj.name = NODE name, but Python's
    // packaging extracts submesh names from MESH name (src/gltf_loader.py:128).
    // Without this remapping, primitive_name matching fails silently.
    const gltfJson = ((gltf as any).parser?.json ?? {}) as {
      meshes?: Array<{ name?: string }>;
      nodes?: Array<{ name?: string; mesh?: number }>;
    };
    const meshNameByIndex = new Map<number, string>();
    (gltfJson.meshes ?? []).forEach((m, i) => {
      meshNameByIndex.set(i, m.name ?? `mesh_${i}`);
    });
    // Build: node.name → mesh.name (resolved via nodes[].mesh index)
    const nodeNameToMeshName = new Map<string, string>();
    for (const node of gltfJson.nodes ?? []) {
      if (node.mesh !== undefined && node.name) {
        const m = meshNameByIndex.get(node.mesh);
        if (m) nodeNameToMeshName.set(node.name, m);
      }
    }

    // 4. Walk glTF scene, find Mesh primitives, match to submesh manifest entries
    const pbrMeshes: PBRMesh[] = [];

    gltf.scene.updateMatrixWorld(true);

    // Build lookup maps for the three match_by strategies.
    // byPrimitiveName resolves the REAL glTF mesh name (not node name).
    const byPrimitiveName = new Map<string, THREE.Mesh[]>();
    const byMaterialName = new Map<string, THREE.Mesh[]>();
    const byMeshIndex = new Map<number, THREE.Mesh[]>();

    let meshIdx = 0;
    gltf.scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;

      // Compute vertex tangents (Mikktspace-style, mirrors Python src/mesh.py:78-141)
      // The source GLB has no TANGENT attribute, so we compute it from geometry.
      if (!obj.geometry.attributes.tangent) {
        computeVertexTangents(obj.geometry);
        console.log(`[PBRPipeline] Computed tangents for mesh ${meshIdx}: ${obj.geometry.attributes.tangent.count} vertices`);
      }

      // Resolve real glTF mesh name by walking up to the top-level node,
      // then mapping node.name → mesh.name via our precomputed lookup.
      let topoParent: THREE.Object3D = obj;
      while (topoParent.parent && topoParent.parent !== gltf.scene) {
        topoParent = topoParent.parent;
      }
      let resolvedMeshName = topoParent.name || `mesh_${meshIdx}`;
      if (nodeNameToMeshName.has(topoParent.name)) {
        resolvedMeshName = nodeNameToMeshName.get(topoParent.name)!;
      }
      console.log(`[PBRPipeline] primitive #${meshIdx}: obj.name="${obj.name}" topoParent.name="${topoParent.name}" → resolvedMeshName="${resolvedMeshName}"`);

      if (!byPrimitiveName.has(resolvedMeshName)) byPrimitiveName.set(resolvedMeshName, []);
      byPrimitiveName.get(resolvedMeshName)!.push(obj);

      // By material name
      const mat = obj.material as THREE.MeshStandardMaterial | THREE.MeshStandardMaterial[];
      const matName = Array.isArray(mat) ? mat[0]?.name : mat?.name;
      if (matName) {
        if (!byMaterialName.has(matName)) byMaterialName.set(matName, []);
        byMaterialName.get(matName)!.push(obj);
      }

      // By mesh index (sequential numbering)
      byMeshIndex.set(meshIdx, [obj]);
      meshIdx++;
    });

    console.log('[PBRPipeline] submesh match lookup summary:');
    console.log('  byPrimitiveName keys:', Array.from(byPrimitiveName.keys()));
    console.log('  byMaterialName keys: ', Array.from(byMaterialName.keys()));
    console.log('  byMeshIndex keys:    ', Array.from(byMeshIndex.keys()));
    console.log('  manifest submeshes:  ', bundle.manifest.submeshes.map(s => `${s.name} (match_by=${s.match_by})`));

    // For each submesh manifest entry, find matching primitive(s)
    for (const submesh of bundle.manifest.submeshes) {
      let primitives: THREE.Mesh[] | undefined;
      switch (submesh.match_by) {
        case 'material_name':
          primitives = byMaterialName.get(submesh.name);
          break;
        case 'mesh_index':
          primitives = byMeshIndex.get(parseInt(submesh.name, 10));
          if (isNaN(parseInt(submesh.name, 10))) {
            console.warn(`submesh "${submesh.name}" has match_by=mesh_index but name is not numeric`);
          }
          break;
        case 'primitive_name':
        default:
          primitives = byPrimitiveName.get(submesh.name);
          break;
      }
      if (!primitives || primitives.length === 0) {
        console.warn(`No glTF primitive found for submesh "${submesh.name}" (match_by=${submesh.match_by})`);
        continue;
      }
      const textureUrls = bundle.submeshTextureUrls[submesh.name];
      if (!textureUrls) {
        console.warn(`No texture URLs for submesh "${submesh.name}"`);
        continue;
      }
      for (const prim of primitives) {
        const pbrMesh = await PBRMesh.fromPrimitive(
          prim,
          submesh,
          textureUrls,
          this.currentEnv,
          brdfLutSize,
        );
        pbrMeshes.push(pbrMesh);
      }
    }

    if (pbrMeshes.length === 0) {
      console.error('[PBRPipeline] FATAL: 0 PBR materials created. Scene will render with glTF default materials.');
    } else {
      console.log(`[PBRPipeline] Created ${pbrMeshes.length} PBR material(s).`);
    }

    // 5. Add glTF scene to render scene
    this.scene.add(gltf.scene);
    this.pbrMeshes = pbrMeshes;

    return { meshes: pbrMeshes, env: this.currentEnv };
  }

  /** Set debug mode on all PBR materials. 0 = off, 1..N = channel. */
  setDebugMode(mode: number): void {
    console.log(`[setDebugMode] mode=${mode}, pbrMeshes.length=${this.pbrMeshes.length}`);
    for (const m of this.pbrMeshes) {
      const mat = m.mesh.material as THREE.ShaderMaterial;
      console.log(`[setDebugMode] mesh=${m.mesh.name}, mat.type=${mat.type}, mat.isShaderMaterial=${mat.isShaderMaterial}, uDebug was=${mat.uniforms.uDebug?.value}`);
      mat.uniforms.uDebug.value = mode;
      mat.needsUpdate = true;
    }
  }

  /** Compute bounding sphere of the loaded scene. */
  getBoundingSphere(): THREE.Sphere {
    const sphere = new THREE.Sphere();
    const box = new THREE.Box3().setFromObject(this.scene);
    if (box.isEmpty()) {
      sphere.set(new THREE.Vector3(0, 0, 0), 1);
    } else {
      box.getBoundingSphere(sphere);
    }
    return sphere;
  }

  /** Get all textures for memory accounting. */
  getTrackedTextures(): THREE.Texture[] {
    const textures: THREE.Texture[] = [];
    if (this.currentEnv) {
      textures.push(this.currentEnv.envMap, this.currentEnv.brdfLut);
    }
    for (const m of this.pbrMeshes) {
      const mat = m.mesh.material as THREE.ShaderMaterial;
      textures.push(
        mat.uniforms.uBaseColor.value,
        mat.uniforms.uRoughness.value,
        mat.uniforms.uMetallic.value,
        mat.uniforms.uNormalMap.value,
      );
    }
    return textures;
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  private disposeScene(): void {
    for (const m of this.pbrMeshes) {
      m.dispose();
    }
    this.pbrMeshes = [];
    if (this.currentEnv) {
      this.currentEnv.dispose();
      this.currentEnv = null;
    }
    // Clear scene children
    while (this.scene.children.length > 0) {
      const child = this.scene.children[0];
      this.scene.remove(child);
    }
  }

  dispose(): void {
    this.disposeScene();
    this.renderer.dispose();
  }
}
