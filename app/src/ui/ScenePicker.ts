import type { SceneIndex } from '../types/manifest';

/**
 * Top toolbar: scene dropdown + drag-drop .zip support.
 */
export class ScenePicker {
  private readonly element: HTMLDivElement;
  private readonly selectEl: HTMLSelectElement;
  private readonly errorToast: HTMLDivElement;
  private scenes: SceneIndex = [];

  /** Fired when user picks a preset scene; receives the scene file URL. */
  onSceneSelect: ((fileUrl: string) => void) | null = null;
  /** Fired when user drops a .zip file; receives the File. */
  onZipDrop: ((file: File) => void) | null = null;

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.style.cssText = `
      position: absolute;
      top: 12px;
      left: 12px;
      padding: 8px 12px;
      background: rgba(0, 0, 0, 0.55);
      color: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 13px;
      border-radius: 6px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 10px;
    `;
    parent.appendChild(this.element);

    const label = document.createElement('span');
    label.textContent = 'Scene:';
    label.style.opacity = '0.7';
    this.element.appendChild(label);

    this.selectEl = document.createElement('select');
    this.selectEl.style.cssText = `
      background: rgba(255, 255, 255, 0.1);
      color: white;
      border: 1px solid rgba(255, 255, 255, 0.2);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 13px;
    `;
    this.selectEl.addEventListener('change', () => {
      const url = this.selectEl.value;
      if (url && this.onSceneSelect) this.onSceneSelect(url);
    });
    this.element.appendChild(this.selectEl);

    // Hint text
    const hint = document.createElement('span');
    hint.textContent = '· or drop .zip';
    hint.style.opacity = '0.5';
    hint.style.fontSize = '11px';
    this.element.appendChild(hint);

    // Drag-drop overlay (whole window)
    this.errorToast = document.createElement('div');
    this.errorToast.style.cssText = `
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      padding: 10px 16px;
      background: rgba(220, 50, 50, 0.9);
      color: white;
      font-family: -apple-system, sans-serif;
      font-size: 13px;
      border-radius: 6px;
      display: none;
      z-index: 100;
    `;
    parent.appendChild(this.errorToast);

    this.setupDragDrop(parent);
  }

  /** Expose the loaded scene list for the App orchestrator. */
  get sceneList(): SceneIndex {
    return this.scenes;
  }

  /** Load the preset scene index. */
  async loadSceneIndex(indexUrl: string): Promise<void> {
    try {
      const resp = await fetch(indexUrl);
      if (!resp.ok) {
        console.warn(`scenes_index.json not found at ${indexUrl}`);
        return;
      }
      this.scenes = (await resp.json()) as SceneIndex;
      this.renderOptions();
    } catch (err) {
      console.warn('Failed to load scene index:', err);
    }
  }

  private renderOptions(): void {
    this.selectEl.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '-- select --';
    this.selectEl.appendChild(placeholder);

    for (const scene of this.scenes) {
      const opt = document.createElement('option');
      opt.value = scene.file;
      const psnr = scene.psnr_db !== null ? ` (${scene.psnr_db.toFixed(1)} dB)` : '';
      opt.textContent = `${scene.name}${psnr}`;
      this.selectEl.appendChild(opt);
    }
  }

  private setupDragDrop(target: HTMLElement): void {
    let dragCounter = 0;

    target.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      target.style.background = 'rgba(0, 255, 136, 0.1)';
    });

    target.addEventListener('dragleave', () => {
      dragCounter--;
      if (dragCounter === 0) {
        target.style.background = '';
      }
    });

    target.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    target.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      target.style.background = '';

      const files = e.dataTransfer?.files;
      if (!files || files.length === 0) return;

      const file = files[0];
      if (!file.name.endsWith('.zip')) {
        this.showError('Please drop a .zip file');
        return;
      }
      if (this.onZipDrop) this.onZipDrop(file);
    });
  }

  showError(message: string): void {
    this.errorToast.textContent = message;
    this.errorToast.style.display = 'block';
    setTimeout(() => {
      this.errorToast.style.display = 'none';
    }, 4000);
  }

  dispose(): void {
    this.element.remove();
    this.errorToast.remove();
  }
}
