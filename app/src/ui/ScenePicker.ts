import type { SceneIndex } from '../types/manifest';

/**
 * Top toolbar: custom scene dropdown + drag-drop .zip support.
 *
 * Uses a custom dropdown (button + popover listbox) instead of a native
 * <select> to guarantee consistent styling across all browsers.
 */
export class ScenePicker {
  private readonly container: HTMLElement;
  private readonly element: HTMLDivElement;
  private readonly triggerBtn: HTMLButtonElement;
  private readonly triggerText: HTMLSpanElement;
  private readonly dropdownEl: HTMLDivElement;
  private readonly errorToast: HTMLDivElement;
  private readonly dropZone: HTMLDivElement;
  private scenes: SceneIndex = [];
  private isOpen = false;
  private selectedIndex = -1;
  private highlightedIndex = -1;

  /** Fired when user picks a preset scene; receives the scene file URL. */
  onSceneSelect: ((fileUrl: string) => void) | null = null;
  /** Fired when user drops a .zip file; receives the File. */
  onZipDrop: ((file: File) => void) | null = null;

  constructor(parent: HTMLElement) {
    this.container = parent;

    // ---- Toolbar ----
    this.element = document.createElement('div');
    this.element.className = 'viewer-toolbar';

    const label = document.createElement('span');
    label.className = 'viewer-toolbar-label';
    label.textContent = 'Scene:';
    this.element.appendChild(label);

    // Dropdown wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'viewer-dropdown-wrapper';

    // Trigger button
    this.triggerBtn = document.createElement('button');
    this.triggerBtn.className = 'viewer-dropdown-trigger';
    this.triggerBtn.type = 'button';
    this.triggerBtn.setAttribute('aria-haspopup', 'listbox');
    this.triggerBtn.setAttribute('aria-expanded', 'false');
    this.triggerBtn.setAttribute('aria-label', 'Select scene');

    this.triggerText = document.createElement('span');
    this.triggerText.className = 'viewer-dropdown-trigger-text';
    this.triggerText.textContent = '-- select --';
    this.triggerBtn.appendChild(this.triggerText);

    // Chevron SVG
    const ns = 'http://www.w3.org/2000/svg';
    const chevron = document.createElementNS(ns, 'svg');
    chevron.setAttribute('class', 'viewer-dropdown-chevron');
    chevron.setAttribute('width', '10');
    chevron.setAttribute('height', '6');
    chevron.setAttribute('viewBox', '0 0 10 6');
    chevron.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', 'M1 1l4 4 4-4');
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '1.5');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    chevron.appendChild(path);
    this.triggerBtn.appendChild(chevron);

    this.triggerBtn.addEventListener('click', () => this.toggleDropdown());
    wrapper.appendChild(this.triggerBtn);

    // Dropdown listbox
    this.dropdownEl = document.createElement('div');
    this.dropdownEl.className = 'viewer-dropdown';
    this.dropdownEl.role = 'listbox';
    this.dropdownEl.setAttribute('aria-label', 'Scenes');
    this.dropdownEl.id = 'scene-dropdown';
    this.dropdownEl.setAttribute('hidden', '');
    this.dropdownEl.addEventListener('keydown', (e) => this.handleKeydown(e));
    wrapper.appendChild(this.dropdownEl);

    this.element.appendChild(wrapper);

    // Hint
    const hint = document.createElement('span');
    hint.className = 'viewer-toolbar-hint';
    hint.textContent = 'drop .zip';
    hint.title = 'Drop a .zip file anywhere to load it';
    this.element.appendChild(hint);

    parent.appendChild(this.element);

    // ---- Error toast ----
    this.errorToast = document.createElement('div');
    this.errorToast.className = 'viewer-error-toast';
    this.errorToast.role = 'alert';
    parent.appendChild(this.errorToast);

    // ---- Drop zone visual overlay ----
    this.dropZone = document.createElement('div');
    this.dropZone.className = 'viewer-drop-zone';
    const dzText = document.createElement('span');
    dzText.className = 'viewer-drop-zone-text';
    dzText.textContent = 'Drop .zip to load';
    this.dropZone.appendChild(dzText);
    parent.appendChild(this.dropZone);

    this.setupDragDrop();
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

  /** Programmatically select a scene by its URL. Updates trigger text and selected state. */
  selectSceneByUrl(url: string): void {
    const index = this.scenes.findIndex((s) => s.file === url);
    if (index < 0) return;
    this.selectedIndex = index;
    this.triggerText.textContent = this.scenes[index].name;
    this.dropdownEl.querySelectorAll('.viewer-dropdown-option').forEach((el, i) => {
      el.setAttribute('aria-selected', String(i === index));
    });
  }

  // ---- Internal helpers ----

  private renderOptions(): void {
    if (this.isOpen) this.closeDropdown();
    this.dropdownEl.innerHTML = '';

    if (this.scenes.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'viewer-dropdown-option viewer-dropdown-option--empty';
      empty.textContent = 'No scenes loaded';
      empty.setAttribute('aria-disabled', 'true');
      this.dropdownEl.appendChild(empty);
      return;
    }

    for (let i = 0; i < this.scenes.length; i++) {
      const scene = this.scenes[i];
      const opt = document.createElement('div');
      opt.className = 'viewer-dropdown-option';
      opt.role = 'option';
      opt.id = `scene-opt-${i}`;
      opt.dataset.index = String(i);
      opt.tabIndex = -1;
      opt.setAttribute('aria-selected', String(i === this.selectedIndex));

      const psnr =
        scene.psnr_db !== null ? ` (${scene.psnr_db.toFixed(1)} dB)` : '';
      opt.textContent = `${scene.name}${psnr}`;

      opt.addEventListener('click', () => this.selectOption(i));
      opt.addEventListener('mouseenter', () => this.highlightOption(i));

      this.dropdownEl.appendChild(opt);
    }
  }

  private toggleDropdown(): void {
    if (this.isOpen) this.closeDropdown();
    else this.openDropdown();
  }

  private openDropdown(): void {
    if (this.scenes.length === 0) return;
    if (this.isOpen) return;

    this.isOpen = true;
    this.highlightedIndex = this.selectedIndex >= 0 ? this.selectedIndex : 0;
    this.triggerBtn.setAttribute('aria-expanded', 'true');
    this.dropdownEl.removeAttribute('hidden');

    this.updateHighlight();
    this.focusOption(this.highlightedIndex);

    // Defer click-outside listener to avoid the opening click itself
    requestAnimationFrame(() => {
      document.addEventListener('click', this.handleClickOutside);
    });
  }

  private closeDropdown(): void {
    if (!this.isOpen) return;
    this.isOpen = false;
    this.triggerBtn.setAttribute('aria-expanded', 'false');
    this.dropdownEl.setAttribute('hidden', '');
    document.removeEventListener('click', this.handleClickOutside);
    this.triggerBtn.focus();
  }

  private handleClickOutside = (e: MouseEvent): void => {
    if (
      !this.dropdownEl.contains(e.target as Node) &&
      !this.triggerBtn.contains(e.target as Node)
    ) {
      this.closeDropdown();
    }
  };

  private selectOption(index: number): void {
    if (index < 0 || index >= this.scenes.length) return;
    const scene = this.scenes[index];
    this.selectedIndex = index;
    this.triggerText.textContent = scene.name;

    this.dropdownEl.querySelectorAll('.viewer-dropdown-option').forEach((el, i) => {
      el.setAttribute('aria-selected', String(i === index));
    });

    this.closeDropdown();
    if (this.onSceneSelect) this.onSceneSelect(scene.file);
  }

  private highlightOption(index: number): void {
    this.highlightedIndex = index;
    this.updateHighlight();
  }

  private updateHighlight(): void {
    this.dropdownEl.querySelectorAll('.viewer-dropdown-option').forEach((el) => {
      el.classList.remove('viewer-dropdown-option--highlighted');
    });
    const target = this.dropdownEl.querySelector<HTMLElement>(
      `[data-index="${this.highlightedIndex}"]`,
    );
    if (target) target.classList.add('viewer-dropdown-option--highlighted');
  }

  private focusOption(index: number): void {
    const el = this.dropdownEl.querySelector<HTMLElement>(
      `[data-index="${index}"]`,
    );
    el?.focus();
  }

  private handleKeydown(e: KeyboardEvent): void {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (this.highlightedIndex < this.scenes.length - 1) {
          this.highlightOption(this.highlightedIndex + 1);
          this.focusOption(this.highlightedIndex);
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (this.highlightedIndex > 0) {
          this.highlightOption(this.highlightedIndex - 1);
          this.focusOption(this.highlightedIndex);
        }
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (this.highlightedIndex >= 0) {
          this.selectOption(this.highlightedIndex);
        }
        break;
      case 'Escape':
        e.preventDefault();
        this.closeDropdown();
        break;
      case 'Tab':
        this.closeDropdown();
        break;
      case 'Home':
        e.preventDefault();
        if (this.scenes.length > 0) {
          this.highlightOption(0);
          this.focusOption(0);
        }
        break;
      case 'End':
        e.preventDefault();
        if (this.scenes.length > 0) {
          this.highlightOption(this.scenes.length - 1);
          this.focusOption(this.scenes.length - 1);
        }
        break;
    }
  }

  private setupDragDrop(): void {
    let dragCounter = 0;

    const show = () => this.dropZone.classList.add('viewer-drop-zone--active');
    const hide = () =>
      this.dropZone.classList.remove('viewer-drop-zone--active');

    this.container.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      show();
    });

    this.container.addEventListener('dragleave', () => {
      dragCounter--;
      if (dragCounter === 0) hide();
    });

    this.container.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    this.container.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      hide();

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
    this.errorToast.classList.add('viewer-error-toast--visible');
    setTimeout(() => {
      this.errorToast.classList.remove('viewer-error-toast--visible');
    }, 4000);
  }

  dispose(): void {
    document.removeEventListener('click', this.handleClickOutside);
    this.element.remove();
    this.errorToast.remove();
    this.dropZone.remove();
  }
}
