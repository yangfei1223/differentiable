import { defineConfig } from 'vite';
import { glslIncludePlugin } from './src/vite/glsl-plugin';
import path from 'node:path';

export default defineConfig({
  // Dev mode: serve ../output so /scenes/*.zip is reachable
  publicDir: path.resolve(__dirname, '../output'),
  plugins: [glslIncludePlugin()],
  server: {
    open: true,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
