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
 *
 * Uses FloatType (RGBA32F). WebGL2's auto-generateMipmap is unreliable for
 * float textures across drivers, so we manually build the mip chain in JS
 * (box-filter average) and assign to texture.mipmaps. This guarantees the
 * mipmap pyramid has the correct HDR-aware color values that match Python's
 * nvdiffrast linear-mipmap-linear sampling.
 */
function loadHdrEnvMap(url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new RGBELoader();
    loader.setDataType(THREE.FloatType); // RGBA32F
    loader.load(
      url,
      (texture) => {
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.RepeatWrapping; // equirect wraps horizontally
        texture.wrapT = THREE.RepeatWrapping; // match Python nvdiffrast boundary_mode="wrap"
        // RGBELoader produces linear data already; ensure colorSpace is NoColorSpace
        // so Three.js doesn't apply any further sRGB decode.
        texture.colorSpace = THREE.NoColorSpace;

        // Manually build mipmap chain (box filter) — WebGL's auto-glGenerateMipmap
        // is unreliable for RGBA32F across drivers.
        const mipmaps = buildFloatMipmapChain(texture.image.data as unknown as Float32Array, texture.image.width, texture.image.height);
        texture.mipmaps = mipmaps.map((data, i) => ({
          data,
          width: Math.max(1, texture.image.width >> i),
          height: Math.max(1, texture.image.height >> i),
        }));
        texture.generateMipmaps = false; // we provide mipmaps manually
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load HDR env map ${url}: ${err}`)),
    );
  });
}

/**
 * Build a box-filtered mipmap chain for a FloatType RGBA texture.
 * Returns array of Float32Array: [mip0_data, mip1_data, ...]
 * where mip0 is the original image and each subsequent entry is half the
 * resolution of the previous (averaging 2x2 blocks).
 *
 * Wrapping is REPEAT in both dimensions (matches env_map.py boundary_mode="wrap").
 */
function buildFloatMipmapChain(
  data: Float32Array,
  width: number,
  height: number,
): Float32Array[] {
  const chain: Float32Array[] = [data];
  let w = width;
  let h = height;
  let current = data;
  while (w > 1 && h > 1) {
    const nw = Math.max(1, w >> 1);
    const nh = Math.max(1, h >> 1);
    const next = new Float32Array(nw * nh * 4);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        // Average 2x2 block from current level (with REPEAT wrap)
        let r = 0, g = 0, b = 0, a = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const sx = (x * 2 + dx) % w;
            const sy = (y * 2 + dy) % h;
            const idx = (sy * w + sx) * 4;
            r += current[idx];
            g += current[idx + 1];
            b += current[idx + 2];
            a += current[idx + 3];
          }
        }
        const o = (y * nw + x) * 4;
        next[o] = r / 4;
        next[o + 1] = g / 4;
        next[o + 2] = b / 4;
        next[o + 3] = a / 4;
      }
    }
    chain.push(next);
    current = next;
    w = nw;
    h = nh;
  }
  return chain;
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
        texture.wrapT = THREE.RepeatWrapping; // match Python nvdiffrast boundary_mode="wrap"
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load texture ${url}: ${err}`)),
    );
  });
}
