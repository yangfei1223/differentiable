import * as THREE from 'three';

/**
 * Loads and configures the environment map + BRDF LUT textures.
 *
 * The env map is loaded as sRGB (matching Python's export of clamped [0,1] PNG),
 * then Three.js converts to linear for shader use.
 * Mipmaps are auto-generated so textureLod can sample prefiltered specular.
 */
export class Environment {
  /** Equirectangular env map texture with mipmaps. */
  readonly envMap: THREE.Texture;
  /** BRDF LUT texture (RG channels). */
  readonly brdfLut: THREE.Texture;
  /** Log2 size of the BRDF LUT (square) — used in align_corners UV correction. */
  readonly brdfLutSize: number;
  /** floor(log2(max(envH, envW))) — max mip level for textureLod. */
  readonly maxEnvMip: number;
  /** Equals maxEnvMip — used as the diffuse irradiance mip bias. */
  readonly diffuseMipBias: number;

  private constructor(
    envMap: THREE.Texture,
    brdfLut: THREE.Texture,
    brdfLutSize: number,
    maxEnvMip: number,
  ) {
    this.envMap = envMap;
    this.brdfLut = brdfLut;
    this.brdfLutSize = brdfLutSize;
    this.maxEnvMip = maxEnvMip;
    this.diffuseMipBias = maxEnvMip;
  }

  /**
   * Build Environment from blob URLs.
   *
   * @param envMapUrl Blob URL for env_map.png
   * @param brdfLutUrl Blob URL for brdf_lut.png
   */
  static async fromUrls(envMapUrl: string, brdfLutUrl: string): Promise<Environment> {
    const envMap = await loadTexture(envMapUrl, THREE.SRGBColorSpace, true);
    const brdfLut = await loadTexture(brdfLutUrl, THREE.LinearSRGBColorSpace, false);

    // Detect BRDF LUT size from the loaded image (assume square)
    const brdfLutSize = brdfLut.image.width || 256;

    // Compute max mip from env map dimensions (after image loads)
    const maxDim = Math.max(envMap.image.width, envMap.image.height);
    const maxEnvMip = Math.floor(Math.log2(maxDim));

    return new Environment(envMap, brdfLut, brdfLutSize, maxEnvMip);
  }

  dispose(): void {
    this.envMap.dispose();
    this.brdfLut.dispose();
  }
}

/**
 * Load a PNG texture from a URL with specified color space and mipmap option.
 */
function loadTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
  generateMipmaps: boolean,
): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = colorSpace;
        texture.generateMipmaps = generateMipmaps;
        texture.minFilter = generateMipmaps
          ? THREE.LinearMipmapLinearFilter
          : THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.RepeatWrapping; // equirect wraps horizontally
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load texture ${url}: ${err}`)),
    );
  });
}
