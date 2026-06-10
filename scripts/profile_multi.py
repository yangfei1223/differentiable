"""Profile multi-mesh rendering speed — measure each stage."""
import time
import torch
from src.config import load_config
from src.mesh import load_mesh, MultiMeshData
from src.dataset import GTDataset
from src.renderer import DifferentiableRenderer
from src.shading import create_shading_model

cfg = load_config("configs/train_pbr_piano_multi.yaml")
device = "cuda"

# Load and prepare
mesh = load_mesh(cfg.data.mesh_path)
assert isinstance(mesh, MultiMeshData), f"Expected MultiMeshData, got {type(mesh)}"

dataset = GTDataset(gt_dir=cfg.data.gt_dir, camera_path=cfg.data.camera_path)
num_views = len(dataset)
batch_size = cfg.training.batch_size

model = create_shading_model(cfg.render_mode, cfg)
model.init_textures(cfg.texture.base_resolution, submesh_names=[s.name for s in mesh.submeshes])

# Build renderers
renderers = {}
for sub in mesh.submeshes:
    v, f, uv, uvi, n, ni, t, bt = sub.to_torch()
    renderers[sub.name] = DifferentiableRenderer(
        vertices=v, faces=f, uvs=uv, uv_idx=uvi,
        normals=n, normal_idx=ni, tangents=t, bitangents=bt,
        resolution=cfg.texture.base_resolution, device=device,
    )

submesh_names = [s.name for s in mesh.submeshes]
resolutions = [512, 1024, 2048]

for res in resolutions:
    print(f"\n{'='*60}")
    print(f"Resolution: {res}x{res}")
    print(f"{'='*60}")

    # Rebuild renderers at this resolution
    for sub in mesh.submeshes:
        v, f, uv, uvi, n, ni, t, bt = sub.to_torch()
        renderers[sub.name] = DifferentiableRenderer(
            vertices=v, faces=f, uvs=uv, uv_idx=uvi,
            normals=n, normal_idx=ni, tangents=t, bitangents=bt,
            resolution=res, device=device,
        )
    model.init_textures(res, submesh_names=submesh_names)

    # Warmup
    img_np, camera = dataset[0]
    for _ in range(2):
        for sub_name in submesh_names:
            renderers[sub_name].rasterize_and_interpolate(camera)

    # --- Profile: rasterization + shade ---
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        for sub_name in submesh_names:
            rast, texc, wpos, inorm, vdir, tang, btang = renderers[sub_name].rasterize_and_interpolate(camera)
            model.shade_submesh(sub_name, rast, texc, wpos, inorm, vdir, camera, res, tang, btang)
            torch.cuda.synchronize()
    t_rast_shade = (time.time() - t0) / 10 / len(submesh_names)
    print(f"  Rasterize+Shade (per submesh): {t_rast_shade*1000:.1f} ms")

    # --- Profile: rasterize only ---
    t0 = time.time()
    for _ in range(10):
        for sub_name in submesh_names:
            rast, texc, wpos, inorm, vdir, tang, btang = renderers[sub_name].rasterize_and_interpolate(camera)
            torch.cuda.synchronize()
    t_rast = (time.time() - t0) / 10 / len(submesh_names)
    print(f"  Rasterize only (per submesh):    {t_rast*1000:.1f} ms")

    # --- Profile: shade only ---
    rast, texc, wpos, inorm, vdir, tang, btang = renderers[submesh_names[0]].rasterize_and_interpolate(camera)
    t0 = time.time()
    for _ in range(10):
        model.shade_submesh(submesh_names[0], rast, texc, wpos, inorm, vdir, camera, res, tang, btang)
        torch.cuda.synchronize()
    t_shade = (time.time() - t0) / 10
    print(f"  Shade only (per call):           {t_shade*1000:.1f} ms")

    total_per_submesh = t_rast_shade
    total_all_submeshes = total_per_submesh * len(submesh_names)
    print(f"  All 6 submeshes (raster+shade):  {total_all_submeshes*1000:.1f} ms")
    print(f"  Per epoch (4 views × 6 subs):    {total_all_submeshes*4000:.0f} ms")

print("\nDone.")
