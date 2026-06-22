import type { Plugin } from 'vite';
import path from 'node:path';
import fs from 'node:fs';

/**
 * Vite plugin that resolves `#include "file.glsl"` directives in .glsl files
 * imported with `?raw` suffix. Inlines the included file content recursively.
 *
 * Usage in TS:
 *   import fragSrc from '../shaders/pbr.frag?raw';
 *
 * In pbr.frag:
 *   #include "common.glsl"
 */
export function glslIncludePlugin(): Plugin {
  const includeRegex = /^#include\s+"([^"]+)"\s*$/gm;

  function resolveIncludes(source: string, dir: string, depth = 0): string {
    if (depth > 8) {
      throw new Error(`GLSL include depth exceeded 8 (circular include?) in ${dir}`);
    }
    return source.replace(includeRegex, (_match, filename) => {
      const includePath = path.resolve(dir, filename);
      const includeSource = fs.readFileSync(includePath, 'utf-8');
      const includeDir = path.dirname(includePath);
      return resolveIncludes(includeSource, includeDir, depth + 1);
    });
  }

  return {
    name: 'glsl-include-resolver',
    enforce: 'pre',
    // Vite's built-in `?raw` handler short-circuits before transform runs,
    // so we intercept `load` (which fires before the built-in ?raw handler
    // when enforce:'pre') and resolve includes inline.
    async load(id) {
      const cleaned = id.replace(/\?.*$/, '');
      const isGlsl = /\.(glsl|vert|frag|vs|fs)$/.test(cleaned);
      const hasRaw = /\?raw(&|$)/.test(id);
      if (!isGlsl || !hasRaw) return null;
      const source = await fs.promises.readFile(cleaned, 'utf-8');
      const resolved = resolveIncludes(source, path.dirname(cleaned));
      return `export default ${JSON.stringify(resolved)};`;
    },
  };
}
