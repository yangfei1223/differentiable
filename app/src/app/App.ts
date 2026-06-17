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
      const first = this.scenePicker.sceneList;
      if (first && first.length > 0 && first[0].file) {
        this.loadScene(first[0].file);
      } else {
        this.loading.setStatus('Drop a .zip to begin');
      }
    });

    // Start animation loop
    this.animate();
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
