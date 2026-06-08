"""测试 ShadingModel 基类。"""
import pytest
from src.shading.base import ShadingModel


def test_shading_model_has_required_methods():
    required = [
        "parameters", "init_textures", "shade",
        "get_material_texture", "set_material_texture",
        "get_debug_info", "export", "state_dict", "load_state_dict",
    ]
    for name in required:
        assert hasattr(ShadingModel, name), f"Missing method: {name}"


def test_create_shading_model_sh():
    from src.shading import create_shading_model
    from src.config import Config

    cfg = Config()
    model = create_shading_model("sh", cfg)
    assert model is not None
    assert hasattr(model, "parameters")


def test_create_shading_model_pbr():
    from src.shading import create_shading_model
    from src.config import Config

    cfg = Config()
    cfg.render_mode = "pbr"
    model = create_shading_model("pbr", cfg)
    assert model is not None


def test_create_shading_model_invalid():
    from src.shading import create_shading_model
    from src.config import Config

    cfg = Config()
    with pytest.raises(ValueError, match="Unknown render_mode"):
        create_shading_model("invalid", cfg)
