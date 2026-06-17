import { App } from './app/App';

const container = document.getElementById('app');
if (!container) {
  throw new Error('#app element not found in DOM');
}

const app = new App(container);

// Expose for debugging in browser console
(window as unknown as { __app: App }).__app = app;
