import * as THREE from 'three';
import type { SubmeshEntry, SubmeshTextures } from '../types/manifest';
import { Environment } from './Environment';

import vertSrc from '../shaders/pbr.vert?raw';
import fragSrc from '../shaders/pbr.frag?raw';

/**
 * Wraps a single submesh (one glTF primitive) with a PBR ShaderMaterial.
 *
 * Loads 4 material textures (base_color, roughness, metallic, normal_map),
 * builds uniforms, and returns a THREE.Mesh ready for the scene.
 */
export class PBRMesh {
  /** Builds a PBRMesh from a glTF primitive (THREE.Mesh) and submesh manifest entry. */
  static async fromPrimitive(
    primitive: THREE.Mesh,
    submesh: SubmeshEntry,
    textureUrls: SubmeshTextures,
    env: Environment,
    brdfLutSize: number,
  ): Promise<PBRMesh> {
    const material = await PBRMesh.buildMaterial(
      textureUrls,
      env,
      brdfLutSize,
    );

    // Replace the primitive's material with our PBR material
    primitive.material = material;

    return new PBRMesh(primitive, material, submesh);
  }

  private static async buildMaterial(
    textureUrls: SubmeshTextures,
    env: Environment,
    brdfLutSize: number,
  ): Promise<THREE.ShaderMaterial> {
    const [baseColor, roughness, metallic, normalMap] = await Promise.all([
      loadMaterialTexture(textureUrls.base_color, THREE.SRGBColorSpace),
      loadMaterialTexture(textureUrls.roughness, THREE.LinearSRGBColorSpace),
      loadMaterialTexture(textureUrls.metallic, THREE.LinearSRGBColorSpace),
      loadMaterialTexture(textureUrls.normal_map, THREE.LinearSRGBColorSpace),
    ]);

    return new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      // Disable Three.js auto-injection of tonemapping/colorspace chunks —
      // our pbr.frag writes final linear color directly; Three.js post-processing
      // (outputColorSpace=SRGB) will handle linear→sRGB at the renderer level.
      toneMapped: false,
      vertexShader: vertSrc,
      fragmentShader: fragSrc,
      uniforms: {
        uBaseColor: { value: baseColor },
        uRoughness: { value: roughness },
        uMetallic: { value: metallic },
        uNormalMap: { value: normalMap },
        uEnvMap: { value: env.envMap },
        uBRDFLut: { value: env.brdfLut },
        uMaxEnvMip: { value: env.maxEnvMip },
        uDiffuseMipBias: { value: env.diffuseMipBias },
        // Match Python training's GT/video rendering: always skips normal mapping
        // because pbr_logger._export_compare (line 90) and video.py (line 221) call
        // shade() WITHOUT tangents/bitangents → pbr_model.py:78 condition fails.
        // Normal map is only used during training forward (trainer.py:545).
        uNormalMapEnabled: { value: false },
        uBRDFLutSize: { value: new THREE.Vector2(brdfLutSize, brdfLutSize) },
        uDebug: { value: 0 },
      },
    });
  }

  private constructor(
    readonly mesh: THREE.Mesh,
    private readonly material: THREE.ShaderMaterial,
    readonly submesh: SubmeshEntry,
  ) {}

  dispose(): void {
    this.material.uniforms.uBaseColor.value.dispose();
    this.material.uniforms.uRoughness.value.dispose();
    this.material.uniforms.uMetallic.value.dispose();
    this.material.uniforms.uNormalMap.value.dispose();
    this.material.dispose();
  }
}

async function loadMaterialTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = colorSpace;
        texture.generateMipmaps = true;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        // RepeatWrapping so fract(uv) in shader + any minor overflow samples correctly
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load material texture ${url}: ${err}`)),
    );
  });
}
