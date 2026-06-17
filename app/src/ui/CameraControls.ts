import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Wraps OrbitControls with mobile-friendly defaults and fit-to-bounds helper.
 *
 * Touch gestures (handled by OrbitControls natively):
 *   - 1 finger: rotate
 *   - 2 finger: pan / pinch-zoom
 */
export class CameraControls {
  readonly controls: OrbitControls;
  private readonly camera: THREE.PerspectiveCamera;

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera;
    this.controls = new OrbitControls(camera, domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 100;
  }

  /** Frame the camera to fit the given bounding sphere. */
  fitToBoundingSphere(sphere: THREE.Sphere): void {
    const center = sphere.center.clone();
    const radius = Math.max(sphere.radius, 0.001);

    // Position camera at a distance that fits the sphere
    const fov = (this.camera.fov * Math.PI) / 180;
    const distance = radius / Math.sin(fov / 2) * 1.2;

    // Place camera along +Z from center (arbitrary, will be user-adjustable)
    this.camera.position.set(center.x, center.y, center.z + distance);
    this.camera.near = distance / 100;
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();

    this.controls.target.copy(center);
    this.controls.update();
  }

  /** Reset camera to default framing of the bounding sphere. */
  reset(sphere: THREE.Sphere): void {
    this.fitToBoundingSphere(sphere);
  }

  update(): void {
    this.controls.update();
  }

  dispose(): void {
    this.controls.dispose();
  }
}
