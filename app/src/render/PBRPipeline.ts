import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import type { AssetBundle } from '../app/SceneLoader';
import { Environment } from './Environment';
import { PBRMesh } from './PBRMesh';

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
    this.currentEnv = await Environment.fromUrls(bundle.envMapUrl, bundle.brdfLutUrl);

    // 2. Determine BRDF LUT size from manifest default (256) — could be detected from image
    const brdfLutSize = 256;

    // 3. Load GLB
    const gltf = await this.gltfLoader.loadAsync(bundle.glbUrl);

    // 4. Walk glTF scene, find Mesh primitives, match to submesh manifest entries
    const allTextures: THREE.Texture[] = [this.currentEnv.envMap, this.currentEnv.brdfLut];
    const pbrMeshes: PBRMesh[] = [];

    gltf.scene.updateMatrixWorld(true);

    // Track primitive index per mesh name to disambiguate when match_by needs adjustment
    const primitivesByName = new Map<string, THREE.Mesh[]>();

    gltf.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        const meshName = obj.name || `mesh_${primitivesByName.size}`;
        if (!primitivesByName.has(meshName)) {
          primitivesByName.set(meshName, []);
        }
        primitivesByName.get(meshName)!.push(obj);
      }
    });

    // For each submesh manifest entry, find matching primitive(s)
    for (const submesh of bundle.manifest.submeshes) {
      const primitives = primitivesByName.get(submesh.name);
      if (!primitives || primitives.length === 0) {
        console.warn(`No glTF primitive found for submesh "${submesh.name}"`);
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
        // Track material textures for memory accounting
        const mat = pbrMesh.mesh.material as THREE.ShaderMaterial;
        allTextures.push(
          mat.uniforms.uBaseColor.value,
          mat.uniforms.uRoughness.value,
          mat.uniforms.uMetallic.value,
          mat.uniforms.uNormalMap.value,
        );
      }
    }

    // 5. Add glTF scene to render scene
    this.scene.add(gltf.scene);
    this.pbrMeshes = pbrMeshes;

    return { meshes: pbrMeshes, env: this.currentEnv };
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
