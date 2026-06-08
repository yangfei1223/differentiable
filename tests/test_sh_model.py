"""测试 SH 着色模型包装。"""
import torch
from src.config import Config
from src.shading.sh_model import SHShadingModel


def test_sh_model_init():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)
    params = model.parameters()
    assert len(params) == 2


def test_sh_model_state_dict():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)
    state = model.state_dict()
    assert "features_dc" in state
    assert "features_rest" in state
    assert "render_mode" in state
    assert state["render_mode"] == "sh"


def test_sh_model_get_material_texture():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)
    tex = model.get_material_texture()
    n_sh = (cfg.texture.sh_order + 1) ** 2
    assert tex.shape[-1] == n_sh * 3


def test_sh_model_set_material_texture():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)

    # Get texture, modify, set back
    tex = model.get_material_texture()
    tex = tex * 2.0  # arbitrary change
    model.set_material_texture(tex)
    tex2 = model.get_material_texture()
    assert torch.allclose(tex2, tex, atol=1e-5)


def test_sh_model_load_state_dict():
    cfg = Config()
    model1 = SHShadingModel(cfg)
    model1.init_textures(16)

    state = model1.state_dict()
    model2 = SHShadingModel(cfg)
    model2.init_textures(16)
    model2.load_state_dict(state)

    tex1 = model1.get_material_texture()
    tex2 = model2.get_material_texture()
    assert torch.allclose(tex1, tex2)
