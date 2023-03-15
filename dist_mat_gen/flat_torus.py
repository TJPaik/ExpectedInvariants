import numpy as np
import torch


def flat_torus_gen(n_points: int) -> np.ndarray:
    data = torch.rand(n_points, 2)
    _data = torch.rand(0, 2)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            _data = torch.concat([_data, data + torch.tensor([i, j])])
    _distance_matrix = torch.cdist(_data, _data)
    distance_matrix = torch.zeros(n_points, n_points)
    for i in range(n_points):
        for j in range(n_points):
            distance_matrix[i, j] = torch.min(_distance_matrix[i::n_points, j::n_points])
    try:
        assert torch.all(~torch.isnan(distance_matrix))
        assert torch.all(distance_matrix >= 0)
    except AssertionError:
        return flat_torus_gen(n_points)
    return distance_matrix.numpy()
