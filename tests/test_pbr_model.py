"""测试 PBR 着色模型。"""
import torch
from src.config import Config, PBRConfig
from src.shading.pbr_model import PBRShadingModel
from src.shading.base import ShadingModel


def _make_cfg(**overrides) -> Config:
    """创建用于测试的小型 Config（brdf_lut_size=16 加速 LUT 生成）。"""
    pbr = PBRConfig(brdf_lut_size=16)
    cfg = Config(pbr=pbr)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_pbr_model_is_shading_model():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    assert isinstance(model, ShadingModel)


def test_pbr_model_init_textures():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    model.init_textures(32)
    params = model.parameters()
    assert len(params) == 2


def test_pbr_model_mat_texture_shape():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    model.init_textures(32)
    mat = model.get_material_texture()
    assert mat.shape == (1, 32, 32, 5)


def test_pbr_model_state_dict():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    model.init_textures(16)
    state = model.state_dict()
    assert state["render_mode"] == "pbr"
    assert "mat_texture" in state
    assert "env_map" in state


def test_pbr_model_load_state_dict():
    cfg = _make_cfg()
    model1 = PBRShadingModel(cfg)
    model1.init_textures(16)
    state = model1.state_dict()

    model2 = PBRShadingModel(cfg)
    model2.init_textures(16)
    model2.load_state_dict(state)

    mat1 = model1.get_material_texture()
    mat2 = model2.get_material_texture()
    assert torch.allclose(mat1, mat2)


def test_pbr_model_has_shade():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    model.init_textures(16)
    assert hasattr(model, "shade")
    assert callable(model.shade)


def test_pbr_model_brdf_lut_shape():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    assert model.brdf_lut.shape == (16, 16, 2)


def test_pbr_model_set_material_texture():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    model.init_textures(16)

    new_tex = torch.randn(1, 16, 16, 5)
    model.set_material_texture(new_tex)

    retrieved = model.get_material_texture()
    assert torch.allclose(new_tex, retrieved)


def test_pbr_model_debug_info_empty():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    assert model.get_debug_info() == {}


def test_pbr_model_env_map_shape():
    cfg = _make_cfg()
    model = PBRShadingModel(cfg)
    model.init_textures(16)
    env = model.env_map.data.detach().cpu()
    assert env.shape == (1, 64, 128, 3)
