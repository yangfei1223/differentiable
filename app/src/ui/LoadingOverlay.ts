/** Loading spinner overlay with status text. */
export class LoadingOverlay {
  private readonly element: HTMLDivElement;
  private readonly statusEl: HTMLDivElement;

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.style.cssText = `
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      z-index: 100;
    `;
    parent.appendChild(this.element);

    const spinner = document.createElement('div');
    spinner.style.cssText = `
      width: 40px;
      height: 40px;
      border: 3px solid rgba(255, 255, 255, 0.2);
      border-top-color: #00ff88;
      border-radius: 50%;
      animation: ps-spin 0.8s linear infinite;
    `;
    this.element.appendChild(spinner);

    this.statusEl = document.createElement('div');
    this.statusEl.style.cssText = `margin-top: 14px; font-size: 13px; opacity: 0.85;`;
    this.statusEl.textContent = 'Loading...';
    this.element.appendChild(this.statusEl);

    // Inject keyframes if not already present
    if (!document.getElementById('ps-spin-keyframes')) {
      const style = document.createElement('style');
      style.id = 'ps-spin-keyframes';
      style.textContent = `@keyframes ps-spin { to { transform: rotate(360deg); } }`;
      document.head.appendChild(style);
    }
  }

  setStatus(text: string): void {
    this.statusEl.textContent = text;
  }

  show(): void {
    this.element.style.display = 'flex';
  }

  hide(): void {
    this.element.style.display = 'none';
  }

  showError(message: string): void {
    this.statusEl.textContent = `Error: ${message}`;
    this.statusEl.style.color = '#ff6b6b';
    // Auto-hide after 5 seconds
    setTimeout(() => this.hide(), 5000);
  }

  dispose(): void {
    this.element.remove();
  }
}
