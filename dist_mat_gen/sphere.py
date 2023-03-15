import numpy as np
import torch


def sphere_gen(n_points: int, n_dim: int) -> np.ndarray:
    # S^{n_dim}
    n_dim += 1
    data = torch.randn(n_points, n_dim)
    data /= torch.norm(data, dim=1)[:, None]
    distance_matrix = torch.cdist(data, data, )
    distance_matrix = 2 * torch.arcsin(distance_matrix / 2)
    try:
        assert torch.all(~torch.isnan(distance_matrix))
        assert torch.all(distance_matrix >= 0)
    except AssertionError:
        return sphere_gen(n_points, n_dim)
    return distance_matrix.numpy()
