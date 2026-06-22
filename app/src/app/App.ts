import { PBRPipeline } from '../render/PBRPipeline';
import { CameraControls } from '../ui/CameraControls';
import { PerfStats } from '../ui/PerfStats';
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
    this.perfStats = new PerfStats(container);
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

      // Re-fit camera
      const sphere = this.pipeline.getBoundingSphere();
      this.cameraControls.fitToBoundingSphere(sphere);

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
  };

  private resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.pipeline.setSize(w, h);
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    if (this.currentBundle) SceneLoader.dispose(this.currentBundle);
    this.cameraControls.dispose();
    this.perfStats.dispose();
    this.loading.dispose();
    this.scenePicker.dispose();
    this.pipeline.dispose();
  }
}
