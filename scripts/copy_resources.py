import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\yangfei\Code\differentiable")
RESOURCE = ROOT / "resource"

mapping = {
    "helmet_260604":          "helmet_sh",
    "helmet_260604_pbr":      "helmet_pbr",
    "piano_260604":           "piano_sh",
    "piano_260604_pbr":       "piano_pbr",
    "piano_260604_pbr_multi": "piano_pbr_multi",
}

skip = {".pt", "pbr_checkpoint", "sh_texture"}

for src_dir_name, dst_dir_name in mapping.items():
    src = ROOT / "output" / src_dir_name / "epoch2000"
    if not src.exists():
        print(f"SKIP: {src} not found")
        continue
    
    dst = RESOURCE / dst_dir_name
    dst.mkdir(parents=True, exist_ok=True)
    
    n = 0
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        if any(s in f.name for s in skip):
            continue
        
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    
    print(f"{src_dir_name} → {dst_dir_name}: {n} files")

print("\nDone.")
