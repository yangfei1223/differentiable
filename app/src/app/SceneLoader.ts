import JSZip from 'jszip';
import type { Manifest, SubmeshTextures } from '../types/manifest';
import { validateManifest } from '../types/manifest';

/**
 * Result of loading a scene: validated manifest + blob URLs for all assets.
 * Caller is responsible for revoking URLs via dispose().
 */
export interface AssetBundle {
  manifest: Manifest;
  glbUrl: string;
  envMapUrl: string;
  brdfLutUrl: string;
  /** Per-submesh texture URLs, indexed by submesh name. */
  submeshTextureUrls: Record<string, SubmeshTextures>;
  /** All created blob URLs, for cleanup. */
  blobUrls: string[];
}

export type LoadProgressCallback = (stage: string) => void;

/**
 * Load and parse a .zip asset bundle.
 *
 * Accepts either:
 *   - A URL string (fetch + arraybuffer)
 *   - A File/Blob (from drag-drop or file input)
 */
export class SceneLoader {
  static async load(
    source: string | File | Blob,
    onProgress?: LoadProgressCallback,
  ): Promise<AssetBundle> {
    onProgress?.('Fetching archive');
    const data = await toArrayBuffer(source);

    onProgress?.('Unzipping');
    const zip = await JSZip.loadAsync(data);

    // Parse manifest
    onProgress?.('Parsing manifest');
    const manifestFile = zip.file('manifest.json');
    if (!manifestFile) {
      throw new Error('manifest.json not found in archive');
    }
    const manifestJson = await manifestFile.async('text');
    const manifest = validateManifest(JSON.parse(manifestJson));

    // Build blob URLs for each asset
    onProgress?.('Extracting geometry');
    const glbUrl = await extractBlobUrl(zip, manifest.geometry.glb_path);

    onProgress?.('Extracting environment');
    const envMapUrl = await extractBlobUrl(zip, manifest.environment.env_map_path);
    const brdfLutUrl = await extractBlobUrl(zip, manifest.brdf_lut_path);

    onProgress?.('Extracting textures');
    const submeshTextureUrlMap: Record<string, SubmeshTextures> = {};
    for (const sub of manifest.submeshes) {
      const urls: Partial<SubmeshTextures> = {};
      for (const key of ['base_color', 'roughness', 'metallic', 'normal_map'] as const) {
        urls[key] = await extractBlobUrl(zip, sub.textures[key]);
      }
      submeshTextureUrlMap[sub.name] = urls as SubmeshTextures;
    }

    const blobUrls = [
      glbUrl,
      envMapUrl,
      brdfLutUrl,
      ...Object.values(submeshTextureUrlMap).flatMap((t) => Object.values(t)),
    ];

    return {
      manifest,
      glbUrl,
      envMapUrl,
      brdfLutUrl,
      submeshTextureUrls: submeshTextureUrlMap,
      blobUrls,
    };
  }

  /** Revoke all blob URLs created by the loader. */
  static dispose(bundle: AssetBundle): void {
    for (const url of bundle.blobUrls) {
      URL.revokeObjectURL(url);
    }
  }
}

async function toArrayBuffer(source: string | File | Blob): Promise<ArrayBuffer> {
  if (typeof source === 'string') {
    const resp = await fetch(source);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${source}: ${resp.status}`);
    }
    return await resp.arrayBuffer();
  }
  return await source.arrayBuffer();
}

async function extractBlobUrl(zip: JSZip, path: string): Promise<string> {
  const file = zip.file(path);
  if (!file) {
    throw new Error(`File not found in archive: ${path}`);
  }
  const blob = await file.async('blob');
  return URL.createObjectURL(blob);
}
