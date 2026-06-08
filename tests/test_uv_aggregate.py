import torch


def test_per_triangle_render_loss_shape():
    from src.uv.aggregate import per_triangle_render_loss
    pixel_loss = torch.ones(1, 4, 4, 3)
    tri_ids = torch.zeros(1, 4, 4, dtype=torch.int64)
    tri_ids[0, 0:2, :] = 0
    tri_ids[0, 2:4, :] = 1
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    num_faces = 2
    result = per_triangle_render_loss(pixel_loss, tri_ids, mask, num_faces)
    assert result.shape == (2,)


def test_per_triangle_render_loss_values():
    from src.uv.aggregate import per_triangle_render_loss
    pixel_loss = torch.ones(1, 4, 4, 3)
    pixel_loss[0, :2, :] = 0.5
    tri_ids = torch.zeros(1, 4, 4, dtype=torch.int64)
    tri_ids[0, 2:, :] = 1
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    num_faces = 2
    result = per_triangle_render_loss(pixel_loss, tri_ids, mask, num_faces)
    assert abs(result[0].item() - 0.5) < 0.01
    assert abs(result[1].item() - 1.0) < 0.01
