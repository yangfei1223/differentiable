import { PBRPipeline } from '../render/PBRPipeline';
import { CameraControls } from '../ui/CameraControls';
import { PerfStats } from '../ui/PerfStats';
import { CameraInfo } from '../ui/CameraInfo';
import { LoadingOverlay } from '../ui/LoadingOverlay';
import { ScenePicker } from '../ui/ScenePicker';
import { SceneLoader, type AssetBundle } from './SceneLoader';

/**
 * Top-level orchestrator: wires together renderer, UI, camera, animation loop.
 */
export class App {
  private readonly pipeline: PBRPipeline;
  private readonly cameraControls: CameraControls;
  private readonly perfStats: PerfStats;
  private readonly cameraInfo: CameraInfo;
  private readonly loading: LoadingOverlay;
  private readonly scenePicker: ScenePicker;
  private readonly container: HTMLElement;
  private currentBundle: AssetBundle | null = null;
  private animationId = 0;

  constructor(container: HTMLElement) {
    this.container = container;

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width: 100%; height: 100%; display: block;';
    container.appendChild(canvas);

    // Pipeline
    this.pipeline = new PBRPipeline(canvas);
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Camera controls
    this.cameraControls = new CameraControls(this.pipeline.camera, canvas);
    canvas.addEventListener('keydown', (e) => {
      if (e.key === 'r' || e.key === 'R') {
        const sphere = this.pipeline.getBoundingSphere();
        this.cameraControls.reset(sphere);
      }
    });

    // Debug channel cycler: window-level keyboard listener (canvas focus is unreliable)
    const debugLabels = [
      'OFF (final)',
      '1: baseColor',
      '2: roughness',
      '3: metallic',
      '4: normalTS',
      '5: normalW',
      '6: NdotV',
      '7: irradiance',
      '8: prefiltered',
      '9: diffuse',
      '10: specular',
      '11: brdf',
      '12: envUV(0.5,0.5) mip0',
      '13: envUV(0.5,0.5) mip9',
      '14: envUV(vUV) mip0',
      '15: specLod',
      '16: NdotV',
      '17: roughness',
      '18: env at R, specLod',
      '19: env at R, mip 5',
      '20: R direction',
      '21: env at warm spot',
      '22: env at cool spot',
      '23: prefiltered (r^2 * mip)',
      '24: prefiltered ((1-NV)*r*mip)',
      '25: prefiltered mip 5',
      '26: prefiltered mip 6',
      '27: prefiltered mip 7',
      '28: prefiltered mip 8',
    ];
    let debugMode = 0;
    const debugHud = document.createElement('div');
    debugHud.id = 'pbr-debug-hud';
    debugHud.style.cssText = 'position:fixed;left:8px;bottom:8px;padding:8px 12px;background:#000;color:#0f0;font:14px monospace;pointer-events:none;border-radius:4px;z-index:9999;border:1px solid #0f0;';
    debugHud.textContent = `DEBUG: ${debugLabels[debugMode]}  (press 0-9, or backtick to cycle)`;
    document.body.appendChild(debugHud);
    window.addEventListener('keydown', (e) => {
      const key = e.key;
      let newMode: number | null = null;
      if (key >= '0' && key <= '9') {
        newMode = parseInt(key, 10);
      } else if (key === '`' || key === '~') {
        newMode = (debugMode + 1) % debugLabels.length;
      }
      if (newMode !== null) {
        debugMode = newMode;
        this.pipeline.setDebugMode(debugMode);
        debugHud.textContent = `DEBUG: ${debugLabels[debugMode]}`;
        console.log('[PBR DEBUG] mode =', debugMode, debugLabels[debugMode]);
      }
    });

    // UI overlays
    // Right-top overlay stack: perf stats + camera info share one anchor
    const overlayStack = document.createElement('div');
    overlayStack.className = 'viewer-overlay-stack';
    container.appendChild(overlayStack);

    this.perfStats = new PerfStats(overlayStack);
    this.cameraInfo = new CameraInfo(overlayStack);
    this.loading = new LoadingOverlay(container);
    this.loading.show();

    this.scenePicker = new ScenePicker(container);
    this.scenePicker.onSceneSelect = (url) => this.loadScene(url);
    this.scenePicker.onZipDrop = (file) => this.loadScene(file);

    // Load preset scene index
    this.scenePicker.loadSceneIndex('/scenes_index.json').then(() => {
      // Auto-select first scene if available
      const scenes = this.scenePicker.sceneList;
      if (scenes && scenes.length > 0 && scenes[0].file) {
        this.scenePicker.selectSceneByUrl(scenes[0].file);
        this.loadScene(scenes[0].file);
      } else {
        this.loading.setStatus('Drop a .zip to begin');
      }
    });

    // Start animation loop
    this.animate();

    // Expose pipeline globally for debugging (agent-browser eval)
    (window as any).__pipeline = this.pipeline;
  }

  private async loadScene(source: string | File): Promise<void> {
    this.loading.show();
    try {
      // Dispose previous bundle's blob URLs
      if (this.currentBundle) {
        SceneLoader.dispose(this.currentBundle);
        this.currentBundle = null;
      }

      this.loading.setStatus('Loading...');
      this.currentBundle = await SceneLoader.load(source, (stage) => {
        this.loading.setStatus(stage);
      });

      this.loading.setStatus('Building scene');
      await this.pipeline.loadScene(this.currentBundle);

      // Check URL hash for camera override (Blender Z-up coords).
      // Format: #cam=px,py,pz,tx,ty,tz,ux,uy,uz,fov
      const camOverride = this.parseCameraHash();
      if (camOverride) {
        console.log('[App] Applying camera override:', camOverride);
        this.cameraControls.setFromBlenderCamera(camOverride);
      } else {
        // Re-fit camera
        const sphere = this.pipeline.getBoundingSphere();
        this.cameraControls.fitToBoundingSphere(sphere);
      }

      // Offscreen render mode: ?render=SIZE triggers one-shot render at SIZE×SIZE
      // and downloads PNG. Used for AB pixel comparison with Python training output.
      const renderSize = new URLSearchParams(window.location.search).get('render');
      if (renderSize) {
        const size = parseInt(renderSize, 10) || 1024;
        console.log(`[App] Offscreen render mode: ${size}x${size}`);
        await this.performOffscreenRender(size);
      }

      // Track textures for memory accounting
      this.perfStats.trackTextures(this.pipeline.getTrackedTextures());

      this.loading.hide();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('Scene load failed:', err);
      this.loading.showError(message);
      this.scenePicker.showError(message);
    }
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);
    this.cameraControls.update();
    this.pipeline.render();
    this.perfStats.onFrame(this.pipeline.renderer);
    this.cameraInfo.onFrame(this.pipeline.camera, this.cameraControls.controls.target);
  };

  /**
   * One-shot offscreen render at given size, download as PNG.
   * Temporarily resizes the canvas, renders one frame, reads pixels,
   * restores canvas, and triggers a download.
   */
  private async performOffscreenRender(size: number): Promise<void> {
    const pipeline = this.pipeline;
    const canvas = pipeline.renderer.domElement;
    const origW = canvas.width;
    const origH = canvas.height;

    // Set square size
    pipeline.setSize(size, size);
    // Render one frame (no UI overlay, no controls update)
    pipeline.render();

    // Read pixels via renderer
    const gl = pipeline.renderer.getContext();
    const pixels = new Uint8Array(size * size * 4);
    gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Flip Y (WebGL bottom-up → PNG top-down)
    const flipped = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      const srcRow = (size - 1 - y) * size * 4;
      const dstRow = y * size * 4;
      flipped.set(pixels.subarray(srcRow, srcRow + size * 4), dstRow);
    }

    // Restore canvas size
    pipeline.setSize(origW, origH);

    // Create canvas, put image data, export PNG blob, trigger download
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = size;
    tmpCanvas.height = size;
    const ctx = tmpCanvas.getContext('2d')!;
    const imageData = ctx.createImageData(size, size);
    imageData.data.set(flipped);
    ctx.putImageData(imageData, 0, 0);

    tmpCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `render_${size}x${size}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      console.log(`[App] Offscreen render downloaded: render_${size}x${size}.png`);
    }, 'image/png');
  }

  private resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.pipeline.setSize(w, h);
  }

  /**
   * Parse camera override from URL hash.
   * Format: #cam=px,py,pz,tx,ty,tz,ux,uy,uz,fov
   * Coords are Blender Z-up; will be converted to Three.js Y-up.
   * Returns null if no override present.
   */
  private parseCameraHash(): {
    position: [number, number, number];
    look_at: [number, number, number];
    up: [number, number, number];
    fov_deg: number;
  } | null {
    const hash = window.location.hash.slice(1);
    if (!hash.startsWith('cam=')) return null;
    const parts = hash.slice(4).split(',').map(parseFloat);
    if (parts.length !== 10 || parts.some(isNaN)) return null;
    return {
      position: [parts[0], parts[1], parts[2]],
      look_at: [parts[3], parts[4], parts[5]],
      up: [parts[6], parts[7], parts[8]],
      fov_deg: parts[9],
    };
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    if (this.currentBundle) SceneLoader.dispose(this.currentBundle);
    this.cameraControls.dispose();
    this.perfStats.dispose();
    this.cameraInfo.dispose();
    this.loading.dispose();
    this.scenePicker.dispose();
    this.pipeline.dispose();
  }
}
