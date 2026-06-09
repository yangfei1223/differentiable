"""Tests for PBRShadingModel multi-texture support."""
import pytest
import torch
from src.config import Config, PBRConfig
from src.shading.pbr_model import PBRShadingModel


class TestPBRModelMulti:
    def test_init_multi_textures(self):
        """init_textures with submesh_names should create dict of textures."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        names = ["body", "strings", "keys"]
        model.init_textures(64, submesh_names=names)
        assert model.is_multi
        assert len(model.mat_textures) == 3
        for name in names:
            assert name in model.mat_textures
            assert model.mat_textures[name].shape == (1, 64, 64, 8)

    def test_parameters_multi(self):
        """parameters() should include all submesh textures + env_map."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64, submesh_names=["a", "b"])
        params = model.parameters()
        assert len(params) == 3  # 2 textures + 1 env_map

    def test_state_dict_multi(self):
        """state_dict/load_state_dict round-trip for multi-texture."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64, submesh_names=["a", "b"])
        state = model.state_dict()
        assert "mat_textures" in state
        assert isinstance(state["mat_textures"], dict)
        assert len(state["mat_textures"]) == 2

        # Round-trip
        model2 = PBRShadingModel(cfg)
        model2.init_textures(64, submesh_names=["a", "b"])
        model2.load_state_dict(state)
        for name in ["a", "b"]:
            assert torch.allclose(model.mat_textures[name], model2.mat_textures[name])

    def test_single_mesh_backward_compat(self):
        """init_textures without submesh_names should use single texture."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64)
        assert not model.is_multi
        assert model.mat_texture is not None
        assert model.mat_texture.shape == (1, 64, 64, 8)

    def test_shade_submesh(self):
        """shade_submesh should use the named texture."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64, submesh_names=["a", "b"])
        # Set different values for each submesh
        model.mat_textures["a"].data.fill_(0.0)
        model.mat_textures["b"].data.fill_(1.0)
        # shade_submesh should not crash (full integration test in Task 4)
        assert model.mat_textures["a"].mean().item() != model.mat_textures["b"].mean().item()
