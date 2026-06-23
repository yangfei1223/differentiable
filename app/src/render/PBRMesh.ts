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
    flipY: boolean,
  ): Promise<PBRMesh> {
    const material = await PBRMesh.buildMaterial(
      textureUrls,
      env,
      brdfLutSize,
      flipY,
    );

    // Replace the primitive's material with our PBR material
    primitive.material = material;

    return new PBRMesh(primitive, material, submesh);
  }

  private static async buildMaterial(
    textureUrls: SubmeshTextures,
    env: Environment,
    brdfLutSize: number,
    flipY: boolean,
  ): Promise<THREE.ShaderMaterial> {
    const [baseColor, roughness, metallic, normalMap] = await Promise.all([
      loadMaterialTexture(textureUrls.base_color, THREE.SRGBColorSpace, flipY),
      loadMaterialTexture(textureUrls.roughness, THREE.LinearSRGBColorSpace, flipY),
      loadMaterialTexture(textureUrls.metallic, THREE.LinearSRGBColorSpace, flipY),
      loadMaterialTexture(textureUrls.normal_map, THREE.LinearSRGBColorSpace, flipY),
    ]);

    return new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      // Match Python nvdiffrast (no culling) + glTF doubleSided materials.
      // Many piano meshes (keys, frame) have faces whose visible side has
      // a normal pointing away from camera; FrontSide culling would drop
      // them, making the mesh appear partly empty. DoubleSide renders both
      // sides so all visible surfaces show up — matching Python training.
      side: THREE.DoubleSide,
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
  flipY: boolean,
): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = colorSpace;
        // Match Python training: dr.texture(..., boundary_mode="clamp").
        // UVs are already in [0,1] so this rarely matters, but ClampToEdge
        // guarantees edge behavior parity for any UVs slightly outside.
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.generateMipmaps = true;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        // flipY controls whether the image is flipped on GPU upload.
        // - true (default, OpenGL): texture row 0 = bottom; matches GL UV convention.
        // - false: texture row 0 = top; matches nvdiffrast raw tensor data layout.
        // Scenes whose source GLB was authored with V=0 at top (e.g., piano)
        // require false; scenes authored with V=0 at bottom (e.g., helmet)
        // require true. Driven by manifest.material_textures_flip_y.
        texture.flipY = flipY;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load material texture ${url}: ${err}`)),
    );
  });
}
