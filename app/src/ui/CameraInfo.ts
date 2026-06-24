import * as THREE from 'three';

/**
 * Real-time camera pose overlay: shows position, look_at, up, fov
 * in Blender Z-up coordinates (matches cameras.json convention).
 *
 * Three.js Y-up → Blender Z-up: (x, y, z)_t → (x, -z, y)_b
 * (inverse of CameraControls.setFromBlenderCamera transform).
 *
 * Updates every frame so screenshots can be reproduced exactly.
 */
export class CameraInfo {
  private readonly element: HTMLDivElement;
  private readonly posEl: HTMLSpanElement;
  private readonly lookEl: HTMLSpanElement;
  private readonly upEl: HTMLSpanElement;
  private readonly fovEl: HTMLSpanElement;
  private readonly hashEl: HTMLSpanElement;

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.className = 'viewer-camera-info';

    this.element.innerHTML = `
      <div class="viewer-perf-title">Camera (Blender Z-up)</div>
      <div class="viewer-perf-row"><span class="viewer-perf-label">pos</span><span class="viewer-perf-value viewer-camera-mono" id="ci-pos">--</span></div>
      <div class="viewer-perf-row"><span class="viewer-perf-label">look</span><span class="viewer-perf-value viewer-camera-mono" id="ci-look">--</span></div>
      <div class="viewer-perf-row"><span class="viewer-perf-label">up</span><span class="viewer-perf-value viewer-camera-mono" id="ci-up">--</span></div>
      <div class="viewer-perf-row"><span class="viewer-perf-label">fov</span><span class="viewer-perf-value" id="ci-fov">--</span></div>
      <div class="viewer-camera-hash" id="ci-hash" title="URL hash format, click to copy">--</div>
    `;
    this.posEl = this.element.querySelector('#ci-pos')!;
    this.lookEl = this.element.querySelector('#ci-look')!;
    this.upEl = this.element.querySelector('#ci-up')!;
    this.fovEl = this.element.querySelector('#ci-fov')!;
    this.hashEl = this.element.querySelector('#ci-hash')!;

    // Click-to-copy hash
    this.hashEl.style.cursor = 'pointer';
    this.hashEl.style.pointerEvents = 'auto';
    this.hashEl.addEventListener('click', () => {
      const hash = this.hashEl.textContent ?? '';
      navigator.clipboard?.writeText(hash).then(
        () => {
          this.hashEl.classList.add('viewer-camera-copied');
          setTimeout(() => this.hashEl.classList.remove('viewer-camera-copied'), 600);
        },
        () => { /* clipboard blocked; ignore */ },
      );
    });

    parent.appendChild(this.element);
  }

  /** Called every animation frame. */
  onFrame(camera: THREE.PerspectiveCamera, target: THREE.Vector3): void {
    // Three.js Y-up → Blender Z-up: (x, y, z)_t → (x, -z, y)_b
    const px = camera.position.x;
    const py = -camera.position.z;
    const pz = camera.position.y;

    const tx = target.x;
    const ty = -target.z;
    const tz = target.y;

    const ux = camera.up.x;
    const uy = -camera.up.z;
    const uz = camera.up.y;

    const fov = camera.fov;

    this.posEl.textContent = `${fmt(px)}, ${fmt(py)}, ${fmt(pz)}`;
    this.lookEl.textContent = `${fmt(tx)}, ${fmt(ty)}, ${fmt(tz)}`;
    this.upEl.textContent = `${fmt(ux)}, ${fmt(uy)}, ${fmt(uz)}`;
    this.fovEl.textContent = `${fov.toFixed(1)}°`;

    // URL hash format: #cam=px,py,pz,tx,ty,tz,ux,uy,uz,fov
    this.hashEl.textContent =
      `#cam=${fmt(px)},${fmt(py)},${fmt(pz)},${fmt(tx)},${fmt(ty)},${fmt(tz)},` +
      `${fmt(ux)},${fmt(uy)},${fmt(uz)},${fov.toFixed(1)}`;
  }

  dispose(): void {
    this.element.remove();
  }
}

/** Format a number to 3 decimal places, with sign always visible. */
function fmt(n: number): string {
  // Avoid -0.000
  const v = Math.abs(n) < 5e-4 ? 0 : n;
  return v.toFixed(3);
}
