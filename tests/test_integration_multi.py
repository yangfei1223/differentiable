"""Integration test: multi-mesh PBR training on piano."""
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_piano_multi_mesh_training():
    """Multi-mesh piano training should run without errors."""
    from src.config import load_config
    from src.trainer import Trainer
    from src.mesh import MultiMeshData

    cfg = load_config("configs/train_pbr_piano_multi.yaml")
    trainer = Trainer(cfg)
    assert trainer.is_multi
    assert len(trainer.renderers) == 6
    assert isinstance(trainer.multi_mesh, MultiMeshData)

    # Run a few epochs
    trainer.train(output_dir="output/test_piano_multi", checkpoint_every=0)
    assert len(trainer.history["epoch"]) > 0
    assert trainer.history["psnr"][-1] > 0
