import * as THREE from 'three';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

/**
 * Loads and configures the environment map + BRDF LUT textures.
 *
 * The env map can be:
 *   - HDR (.hdr RGBE) — preferred. Preserves training-time HDR values that
 *     PNG would clamp to [0,1], distorting the diffuse/specular color balance.
 *     Python training decodes env_map via softplus: max can reach ~17.
 *   - LDR (.png) — fallback. Lossy. Use only when no HDR available.
 *
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
   * @param envMapUrl Blob URL for env_map.hdr or env_map.png
   * @param brdfLutUrl Blob URL for brdf_lut.png
   * @param isHdr True if env map is HDR (.hdr RGBE format)
   */
  static async fromUrls(
    envMapUrl: string,
    brdfLutUrl: string,
    isHdr: boolean,
  ): Promise<Environment> {
    // HDR (RGBE) loader returns a DataTexture with HalfFloatType data already
    // in linear space. No colorspace conversion needed.
    // LDR PNG path mirrors Python training's interpretation: env_map.png is
    // stored with LINEAR values (decoded softplus directly ×255, no sRGB
    // encoding). Use LinearSRGBColorSpace to avoid incorrect sRGB→linear.
    const envMap = isHdr
      ? await loadHdrEnvMap(envMapUrl)
      : await loadTexture(envMapUrl, THREE.LinearSRGBColorSpace, true);
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
 * Load an HDR (.hdr RGBE) env map. Returns a DataTexture in linear space.
 */
function loadHdrEnvMap(url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new RGBELoader();
    loader.setDataType(THREE.HalfFloatType); // efficient for sampling
    loader.load(
      url,
      (texture) => {
        texture.generateMipmaps = true;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.RepeatWrapping; // equirect wraps horizontally
        texture.wrapT = THREE.ClampToEdgeWrapping;
        // RGBELoader produces linear data already; ensure colorSpace is NoColorSpace
        // so Three.js doesn't apply any further sRGB decode.
        texture.colorSpace = THREE.NoColorSpace;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load HDR env map ${url}: ${err}`)),
    );
  });
}

/**
 * Load a PNG texture (LDR fallback) with specified color space and mipmap option.
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
