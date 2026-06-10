"""Profile full training step — find the bottleneck."""
import time, torch, torch.nn.functional as F
from torch.optim import Adam

from src.config import load_config
from src.mesh import load_mesh, MultiMeshData
from src.dataset import GTDataset
from src.renderer import DifferentiableRenderer
from src.losses import CombinedLoss, tv_loss
from src.shading import create_shading_model

cfg = load_config("configs/train_pbr_piano_multi.yaml")
device = "cuda"

mesh = load_mesh(cfg.data.mesh_path)
assert isinstance(mesh, MultiMeshData)
dataset = GTDataset(gt_dir=cfg.data.gt_dir, camera_path=cfg.data.camera_path)

model = create_shading_model(cfg.render_mode, cfg)
model.init_textures(512, submesh_names=[s.name for s in mesh.submeshes])

renderers = {}
for sub in mesh.submeshes:
    v, f, uv, uvi, n, ni, t, bt = sub.to_torch()
    renderers[sub.name] = DifferentiableRenderer(
        vertices=v, faces=f, uvs=uv, uv_idx=uvi,
        normals=n, normal_idx=ni, tangents=t, bitangents=bt,
        resolution=512, device=device,
    )

submesh_names = [s.name for s in mesh.submeshes]
criterion = CombinedLoss(1.0, 0.2, 0.005)

res = 512

for res in [512, 1024, 2048]:
    # Rebuild at resolution
    model.init_textures(res, submesh_names=submesh_names)
    for sub in mesh.submeshes:
        v, f, uv, uvi, n, ni, t, bt = sub.to_torch()
        renderers[sub.name] = DifferentiableRenderer(
            vertices=v, faces=f, uvs=uv, uv_idx=uvi,
            normals=n, normal_idx=ni, tangents=t, bitangents=bt,
            resolution=res, device=device,
        )

    optimizer = Adam(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(2):
        img_np, camera = dataset[0]
        gt = torch.from_numpy(img_np).unsqueeze(0).to(device)
        for sub_name in submesh_names:
            renderers[sub_name].rasterize_and_interpolate(camera)

    # Full training step profile (avg of 10 steps)
    times = {"render": 0, "gt_prep": 0, "loss": 0, "backward": 0, "optimizer": 0, "total": 0}
    n_steps = 10

    for step in range(n_steps):
        img_np, camera = dataset[step % len(dataset)]
        gt = torch.from_numpy(img_np).unsqueeze(0).to(device)

        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()

        # -- Render --
        rendered = torch.zeros(1, res, res, 3, device=device)
        depth_buf = torch.full((1, res, res), float("inf"), device=device)
        mask = torch.zeros(1, res, res, device=device)
        for sub_name in submesh_names:
            rast, texc, wpos, inorm, vdir, tang, btang = renderers[sub_name].rasterize_and_interpolate(camera)
            rgb_sub, mask_sub = model.shade_submesh(sub_name, rast, texc, wpos, inorm, vdir, camera, res, tang, btang)
            sub_depth = rast[..., 2]
            write = (mask_sub > 0.5) & (sub_depth < depth_buf)
            rendered = torch.where(write.unsqueeze(-1), rgb_sub, rendered)
            depth_buf = torch.where(write, sub_depth, depth_buf)
            mask = torch.max(mask, mask_sub)

        rendered = rendered.flip(1)
        mask = mask.flip(1)
        torch.cuda.synchronize()
        t_render = time.time()

        # -- GT prep --
        gt_hw = gt.permute(0, 1, 2, 3)
        H, W = rendered.shape[1], rendered.shape[2]
        gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
        gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        gt_linear = gt_resized.clamp(0, 1).pow(2.2)
        torch.cuda.synchronize()
        t_gt = time.time()

        # -- Loss --
        tex_dict = model.get_material_texture()
        tex_for_loss = torch.cat(list(tex_dict.values()), dim=0).to(device)
        loss = criterion(rendered, gt_linear, mask, tex_for_loss)
        env_tv = tv_loss(model.env_map.raw) * cfg.pbr.env_tv_weight
        env_decoded = model.env_map.decode()
        env_l2 = (env_decoded ** 2).mean() * cfg.pbr.env_l2_weight
        loss = loss + env_tv + env_l2
        torch.cuda.synchronize()
        t_loss = time.time()

        # -- Backward --
        loss.backward()
        torch.cuda.synchronize()
        t_back = time.time()

        # -- Optimizer --
        optimizer.step()
        torch.cuda.synchronize()
        t_opt = time.time()

        times["render"] += (t_render - t0) * 1000
        times["gt_prep"] += (t_gt - t_render) * 1000
        times["loss"] += (t_loss - t_gt) * 1000
        times["backward"] += (t_back - t_loss) * 1000
        times["optimizer"] += (t_opt - t_back) * 1000
        times["total"] += (t_opt - t0) * 1000

    print(f"\n{'='*60}")
    print(f"Resolution: {res}x{res}  (avg of {n_steps} steps)")
    print(f"{'='*60}")
    for k in ["render", "gt_prep", "loss", "backward", "optimizer", "total"]:
        avg = times[k] / n_steps
        print(f"  {k:12s}: {avg:8.1f} ms")
    print(f"  Per epoch (4 views):      {times['total'] / n_steps * 4:.0f} ms")
    print(f"  Estimated 2000 epochs:    {times['total'] / n_steps * 4 * 2000 / 1000:.0f} s")

print("\nDone.")
