import * as THREE from 'three';

/**
 * Real-time performance overlay: FPS, draw calls, triangle count, texture memory.
 * Updates every 500ms.
 */
export class PerfStats {
  private readonly element: HTMLDivElement;
  private readonly fpsEl: HTMLSpanElement;
  private readonly drawEl: HTMLSpanElement;
  private readonly trisEl: HTMLSpanElement;
  private readonly texEl: HTMLSpanElement;

  private frameCount = 0;
  private lastSampleTime = performance.now();
  private sampleIntervalMs = 500;
  private textures: THREE.Texture[] = [];

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.style.cssText = `
      position: absolute;
      top: 12px;
      right: 12px;
      padding: 10px 14px;
      background: rgba(0, 0, 0, 0.55);
      color: #00ff88;
      font-family: 'SF Mono', Consolas, monospace;
      font-size: 12px;
      border-radius: 6px;
      pointer-events: none;
      z-index: 10;
      line-height: 1.6;
    `;
    parent.appendChild(this.element);

    this.element.innerHTML = `
      <div>FPS: <span id="ps-fps">--</span></div>
      <div>Draw: <span id="ps-draw">--</span></div>
      <div>Tris: <span id="ps-tris">--</span></div>
      <div>Tex: <span id="ps-tex">--</span></div>
    `;
    this.fpsEl = this.element.querySelector('#ps-fps')!;
    this.drawEl = this.element.querySelector('#ps-draw')!;
    this.trisEl = this.element.querySelector('#ps-tris')!;
    this.texEl = this.element.querySelector('#ps-tex')!;
  }

  /** Register textures to track memory usage. */
  trackTextures(textures: THREE.Texture[]): void {
    this.textures = textures;
  }

  /** Called every animation frame. */
  onFrame(renderer: THREE.WebGLRenderer): void {
    this.frameCount++;
    const now = performance.now();
    const elapsed = now - this.lastSampleTime;
    if (elapsed < this.sampleIntervalMs) return;

    const fps = (this.frameCount * 1000) / elapsed;
    this.fpsEl.textContent = fps.toFixed(0);

    const info = renderer.info;
    this.drawEl.textContent = info.render.calls.toString();
    this.trisEl.textContent = formatNumber(info.render.triangles);

    // Estimate texture memory (RGBA = 4 bytes per pixel)
    let bytes = 0;
    for (const tex of this.textures) {
      const img = tex.image as { width?: number; height?: number };
      if (img?.width && img?.height) {
        bytes += img.width * img.height * 4;
        // Include mipmaps (~1.33x)
        if (tex.generateMipmaps) bytes = Math.floor(bytes * 1.33);
      }
    }
    this.texEl.textContent = formatBytes(bytes);

    this.frameCount = 0;
    this.lastSampleTime = now;
  }

  dispose(): void {
    this.element.remove();
  }
}

function formatNumber(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(0) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(0) + ' KB';
  return bytes + ' B';
}
