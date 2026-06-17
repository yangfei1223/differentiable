/**
 * Loading spinner overlay with status text and fade-out transition.
 */
export class LoadingOverlay {
  private readonly element: HTMLDivElement;
  private readonly statusEl: HTMLDivElement;
  private errorTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.className = 'viewer-loading viewer-loading--hidden';

    const spinner = document.createElement('div');
    spinner.className = 'viewer-loading-spinner';
    this.element.appendChild(spinner);

    this.statusEl = document.createElement('div');
    this.statusEl.className = 'viewer-loading-text';
    this.statusEl.textContent = 'Loading...';
    this.element.appendChild(this.statusEl);

    parent.appendChild(this.element);
  }

  setStatus(text: string): void {
    this.statusEl.textContent = text;
    this.statusEl.style.color = '';
  }

  show(): void {
    if (this.errorTimer) {
      clearTimeout(this.errorTimer);
      this.errorTimer = null;
    }
    this.element.classList.remove('viewer-loading--hidden', 'viewer-loading--error');
    this.statusEl.style.color = '';
  }

  hide(): void {
    // Fade out via CSS transition
    this.element.classList.add('viewer-loading--hidden');
  }

  showError(message: string): void {
    this.element.classList.remove('viewer-loading--hidden');
    this.element.classList.add('viewer-loading--error');
    this.statusEl.textContent = `Error: ${message}`;
    this.statusEl.style.color = 'var(--error)';

    // Auto-hide after 5 seconds
    this.errorTimer = setTimeout(() => {
      this.element.classList.add('viewer-loading--hidden');
      // Remove error styling after transition completes
      setTimeout(() => {
        this.element.classList.remove('viewer-loading--error');
        this.statusEl.style.color = '';
      }, 400);
    }, 5000);
  }

  dispose(): void {
    if (this.errorTimer) clearTimeout(this.errorTimer);
    this.element.remove();
  }
}
