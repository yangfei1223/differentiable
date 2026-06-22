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

  /**
   * Set explicit camera from Blender Z-up coords (training cameras.json).
   * Conversion: Blender (X, Y, Z_up) → Three.js (X, Z_up, -Y).
   */
  setFromBlenderCamera(params: {
    position: [number, number, number];
    look_at: [number, number, number];
    up: [number, number, number];
    fov_deg: number;
  }): void {
    const [bx, by, bz] = params.position;
    const [tx, ty, tz] = params.look_at;
    const [ux, uy, uz] = params.up;
    // Blender Z-up to Three.js Y-up: (x, y, z)_b -> (x, z, -y)_t
    this.camera.position.set(bx, bz, -by);
    this.controls.target.set(tx, tz, -ty);
    // Three.js OrbitControls uses camera.up; for full up vector control we'd
    // need to compute a lookAt matrix manually. For now set the up vector.
    this.camera.up.set(ux, uz, -uy);
    this.camera.fov = params.fov_deg;
    this.camera.near = 0.01;
    this.camera.far = 1000;
    this.camera.updateProjectionMatrix();
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
