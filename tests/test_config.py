"""测试配置系统 — render_mode + PBRConfig + UVOptConfig。"""
import tempfile
import pytest
import yaml
from src.config import Config, PBRConfig, UVOptConfig, load_config


def test_default_config_has_render_mode():
    cfg = Config()
    assert cfg.render_mode == "sh"


def test_uv_opt_config_defaults():
    cfg = UVOptConfig()
    assert cfg.enabled is False
    assert cfg.lr == 0.001
    assert cfg.tex_steps_per_uv == 5
    assert cfg.sym_dirichlet_weight == 0.01
    assert cfg.content_aware_weight == 5.0
    assert cfg.area_preserve_weight == 0.1
    assert cfg.start_epoch == 100
    assert cfg.stop_epoch == -1


def test_uv_opt_config_from_yaml(tmp_path):
    data = {
        "render_mode": "pbr",
        "uv_optimization": {
            "enabled": True,
            "lr": 0.002,
            "sym_dirichlet_weight": 0.05,
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = load_config(str(p))
    assert cfg.uv_opt.enabled is True
    assert cfg.uv_opt.lr == 0.002
    assert cfg.uv_opt.sym_dirichlet_weight == 0.05
    assert cfg.uv_opt.area_preserve_weight == 0.1

def test_default_config_has_pbr():
    cfg = Config()
    assert isinstance(cfg.pbr, PBRConfig)
    assert cfg.pbr.env_map_res == [256, 512]
    assert cfg.pbr.n_mip_levels == 5
    assert cfg.pbr.brdf_lut_size == 256
    assert cfg.pbr.env_lr_ratio == 1.0
    assert cfg.pbr.env_tv_weight == 0.0005
    assert cfg.pbr.init_env_map is None


def test_load_config_with_pbr():
    yaml_content = """
render_mode: pbr
pbr:
  env_map_res: [128, 256]
  n_mip_levels: 7
  env_lr_ratio: 0.5
  init_env_map: /path/to/env.hdr
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.render_mode == "pbr"
    assert cfg.pbr.env_map_res == [128, 256]
    assert cfg.pbr.n_mip_levels == 7
    assert cfg.pbr.env_lr_ratio == 0.5
    assert cfg.pbr.init_env_map == "/path/to/env.hdr"


def test_load_config_default_render_mode():
    yaml_content = """
data:
  mesh_path: test.obj
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.render_mode == "sh"
