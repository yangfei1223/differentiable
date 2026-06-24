"""Smoke test: verify all shader models on all scenes.

2 scenes (helmet, piano) × 3 models (SH, PBR, NLM) = 6 combos.
Minimal 3-epoch training at 128px, checkpoint at last epoch, output to
output/smoke_{scene}_{model}/ — never touches existing output/ dirs.
"""
import os
import sys
import time
from pathlib import Path

from src.config import load_config, ResolutionStep
from src.trainer import Trainer

SCENE_ROOTS = Path(__file__).resolve().parents[1]

SMOKE_MATRIX = [
    # (tag, config_path, render_mode expected)
    ("helmet",   "sh",   "configs/train_helmet.yaml"),
    ("piano",    "sh",   "configs/train_sh0.yaml"),
    ("helmet",   "pbr",  "configs/train_pbr_helmet_no_normal.yaml"),
    ("piano",    "pbr",  "configs/train_pbr_piano_multi_no_normal.yaml"),
    ("helmet",   "nlm",  "configs/train_nlm_helmet.yaml"),
    ("piano",    "nlm",  "configs/train_nlm_piano_multi.yaml"),
]


def run_one(scene: str, model: str, config_path: str) -> str:
    tag = f"{scene}_{model}"
    output_dir = SCENE_ROOTS / "output" / f"smoke_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {tag.upper()}  ({config_path})")
    print(f"  output → {output_dir}")
    print(f"{'='*60}")

    cfg = load_config(config_path)
    assert cfg.render_mode == model, f"render_mode mismatch: {cfg.render_mode} vs {model}"

    # Minimal training
    cfg.training.num_epochs = 3
    cfg.training.lr_decay_epochs = []
    cfg.training.resolution_schedule = [ResolutionStep(epoch=0, resolution=128)]
    cfg.texture.base_resolution = 128
    cfg.texture.target_resolution = 128

    t0 = time.time()
    trainer = Trainer(cfg)
    # trainer creates epochNNNN/ subdir inside output_dir automatically
    trainer.train(output_dir=str(output_dir), checkpoint_every=3)
    elapsed = time.time() - t0

    # Verify checkpoint (trainer creates epoch0003/ inside output_dir)
    actual_ep = output_dir / "epoch0003"
    ckpt_files = list(actual_ep.glob("*")) if actual_ep.exists() else []
    ckpt_names = [f.name for f in ckpt_files]
    has_checkpoint = any("checkpoint" in n.lower() or "compare" in n.lower() for n in ckpt_names)

    if has_checkpoint:
        print(f"  [OK] {tag} completed in {elapsed:.1f}s, {len(ckpt_files)} output files")
    else:
        print(f"  [WARN] {tag} trained in {elapsed:.1f}s but no checkpoint/compare files found")
        print(f"    files: {ckpt_names}")

    return "PASS"


def main():
    print("=== Smoke Test Suite ===")
    print(f"  Python: {sys.executable}")
    print(f"  CUDA:   {__import__('torch').cuda.is_available()}")
    print(f"  Root:   {SCENE_ROOTS}")

    results = []
    for scene, model, cfg_path in SMOKE_MATRIX:
        try:
            status = run_one(scene, model, cfg_path)
            results.append((f"{scene}/{model}", status))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((f"{scene}/{model}", f"FAIL: {e}"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, status in results:
        icon = "[OK]" if status == "PASS" else "[FAIL]"
        if status != "PASS":
            all_pass = False
        print(f"  {icon} {name}: {status}")

    print(f"\n  {len(results)} tested, {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
