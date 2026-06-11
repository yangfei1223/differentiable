"""Extract PSNR history from training curves images to get peak PSNR values."""
import torch
import os

checks = {
    "helmet_SH":     ("output/helmet_260604",        "sh"),
    "helmet_PBR":    ("output/helmet_260604_pbr",    "pbr"),
    "piano_SH":      ("output/piano_260604",         "sh"),
    "piano_PBR_1":   ("output/piano_260604_pbr",     "pbr"),
    "piano_PBR_N":   ("output/piano_260604_pbr_multi","pbr"),
}

for name, (dir_path, mode) in checks.items():
    ckpt_path = os.path.join(dir_path, "epoch2000", "pbr_checkpoint.pt")
    sh_path = os.path.join(dir_path, "sh_texture.pt")
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("loss", 0)
        print(f"{name}:  mode={mode}, epoch={epoch}, loss={loss:.4f}")
        
        # Check if we can decode PSNR from the texture/resolution info
        res = ckpt.get("resolution", "?")
        print(f"       resolution={res}")
        
        if mode == "pbr" and "mat_textures" in ckpt:
            names = list(ckpt["mat_textures"].keys())
            print(f"       submeshes={names}")
            
    except Exception as e:
        print(f"{name}:  NOT FOUND ({e})")
