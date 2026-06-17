/**
 * TypeScript interfaces for manifest.json schema.
 * Mirrors Python: scripts/package_runtime_asset.py -> build_manifest
 * Schema version: 1
 */

export type SubmeshMatchBy = 'primitive_name' | 'material_name' | 'mesh_index';

export interface SubmeshTextures {
  base_color: string;
  roughness: string;
  metallic: string;
  normal_map: string;
}

export interface SubmeshEntry {
  name: string;
  match_by: SubmeshMatchBy;
  textures: SubmeshTextures;
}

export interface ManifestGenerator {
  tool: string;
  version: string;
  render_mode: 'pbr';
  epoch: number;
  psnr_db: number | null;
}

export interface ManifestGeometry {
  glb_path: string;
  up_axis: 'Y' | 'Z';
  scale: number;
}

export interface ManifestEnvironment {
  env_map_path: string;
  is_hdr: boolean;
  diffuse_mip_bias: number; // -1 = auto
  specular_mip_levels: number; // -1 = auto
}

export interface Manifest {
  schema_version: 1;
  scene_name: string;
  generator: ManifestGenerator;
  geometry: ManifestGeometry;
  environment: ManifestEnvironment;
  brdf_lut_path: string;
  submeshes: SubmeshEntry[];
}

export interface SceneIndexEntry {
  name: string;
  file: string;
  psnr_db: number | null;
  epoch: number;
}

export type SceneIndex = SceneIndexEntry[];

/**
 * Validate a parsed JSON object against the Manifest schema.
 * Throws Error with descriptive message if invalid.
 */
export function validateManifest(data: unknown): Manifest {
  if (typeof data !== 'object' || data === null) {
    throw new Error('Manifest must be a JSON object');
  }
  const m = data as Record<string, unknown>;
  if (m.schema_version !== 1) {
    throw new Error(`Unsupported schema_version: ${m.schema_version} (expected 1)`);
  }
  if (typeof m.scene_name !== 'string') {
    throw new Error('manifest.scene_name must be a string');
  }
  if (typeof m.brdf_lut_path !== 'string') {
    throw new Error('manifest.brdf_lut_path must be a string');
  }
  if (!Array.isArray(m.submeshes) || m.submeshes.length === 0) {
    throw new Error('manifest.submeshes must be a non-empty array');
  }
  for (let i = 0; i < m.submeshes.length; i++) {
    const s = m.submeshes[i] as Record<string, unknown>;
    if (typeof s.name !== 'string') throw new Error(`submeshes[${i}].name must be string`);
    if (!['primitive_name', 'material_name', 'mesh_index'].includes(s.match_by as string)) {
      throw new Error(`submeshes[${i}].match_by invalid: ${s.match_by}`);
    }
    const t = s.textures as Record<string, unknown>;
    for (const k of ['base_color', 'roughness', 'metallic', 'normal_map']) {
      if (typeof t[k] !== 'string') {
        throw new Error(`submeshes[${i}].textures.${k} must be a string`);
      }
    }
  }
  return data as Manifest;
}
