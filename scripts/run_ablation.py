"""AB 实验：SH order 0 vs SH order 2，各 500 epoch。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import load_config, ResolutionStep
from src.trainer import Trainer


def run_experiment(name, config_path, output_dir):
    cfg = load_config(config_path)
    cfg.training.num_epochs = 500
    cfg.training.lr_decay_epochs = [200, 350, 450]
    cfg.training.resolution_schedule = [
        ResolutionStep(epoch=0, resolution=512),
        ResolutionStep(epoch=250, resolution=1024),
    ]

    print("=" * 60)
    print(f"Experiment: {name}")
    print(f"  SH order: {cfg.texture.sh_order}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    trainer = Trainer(cfg)
    trainer.train(output_dir=output_dir, checkpoint_every=500)

    # Save final texture
    os.makedirs(output_dir, exist_ok=True)
    tex = trainer.get_sh_texture()
    torch.save(tex, os.path.join(output_dir, "sh_texture.pt"))

    # Also save features_dc and features_rest separately
    torch.save(
        {
            "features_dc": trainer.features_dc.data.detach().cpu(),
            "features_rest": trainer.features_rest.data.detach().cpu(),
        },
        os.path.join(output_dir, "sh_features.pt"),
    )

    h = trainer.history
    final_loss = h["loss"][-1]
    final_psnr = h["psnr"][-1]
    recent_psnr = h["psnr"][-5:]

    print(f"\n[{name}] Done.")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Final PSNR: {final_psnr:.2f} dB")
    print(f"  Recent PSNR: {[f'{p:.2f}' for p in recent_psnr]}")
    print()

    return trainer


if __name__ == "__main__":
    base = "output/piano_260604"

    t0 = run_experiment("SH order 0", "configs/train_sh0.yaml", f"{base}/sh0")
    t2 = run_experiment("SH order 2", "configs/default.yaml", f"{base}/sh2")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    h0 = t0.history
    h2 = t2.history
    print(f"SH0: loss={h0['loss'][-1]:.6f}  PSNR={h0['psnr'][-1]:.2f} dB")
    print(f"SH2: loss={h2['loss'][-1]:.6f}  PSNR={h2['psnr'][-1]:.2f} dB")
    print(f"\nOutput:")
    print(f"  {base}/sh0/")
    print(f"  {base}/sh2/")
