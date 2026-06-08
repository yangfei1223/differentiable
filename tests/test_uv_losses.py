import torch


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_sym_dirichlet_identity():
    from src.uv.losses import SymDirichletLoss
    dev = _device()
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device=dev)
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device=dev)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=dev)
    render_loss = torch.tensor([0.5], device=dev)
    loss_fn = SymDirichletLoss()
    loss = loss_fn(uv, verts, faces, render_loss)
    assert loss.item() > 0
    assert loss.item() < 20.0, f"Identity mapping energy too high: {loss.item()}"


def test_sym_dirichlet_flip_penalty():
    from src.uv.losses import SymDirichletLoss
    dev = _device()
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device=dev)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=dev)
    render_loss = torch.tensor([1.0], device=dev)
    uv_flipped = torch.tensor([[0.0, 0.0], [0.5, 0.866], [1.0, 0.0]], device=dev)
    uv_normal = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device=dev)
    loss_fn = SymDirichletLoss()
    loss_flipped = loss_fn(uv_flipped, verts, faces, render_loss)
    loss_normal = loss_fn(uv_normal, verts, faces, render_loss)
    assert loss_flipped.item() > loss_normal.item(), "Flipped UV should have higher energy"


def test_area_preserve_same_area():
    from src.uv.losses import AreaPreserveLoss
    dev = _device()
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device=dev)
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device=dev)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=dev)
    target_areas = torch.tensor([0.433], device=dev)
    loss_fn = AreaPreserveLoss()
    loss = loss_fn(uv, verts, faces, target_areas)
    assert loss.item() < 0.01, f"Same area should give ~0 loss: {loss.item()}"


def test_area_preserve_gradient():
    from src.uv.losses import AreaPreserveLoss
    dev = _device()
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device=dev, requires_grad=True)
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device=dev)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=dev)
    target_areas = torch.tensor([0.1], device=dev)
    loss_fn = AreaPreserveLoss()
    loss = loss_fn(uv, verts, faces, target_areas)
    loss.backward()
    assert uv.grad is not None
    assert uv.grad.norm().item() > 0
