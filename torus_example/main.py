import matplotlib.pyplot as plt
import numpy as np
import torch

from top_curve import Betti_curve


def plot_torus(precision, c, a):
    U = np.linspace(0, 2 * np.pi, precision)
    V = np.linspace(0, 2 * np.pi, precision)
    U, V = np.meshgrid(U, V)
    X = (c + a * np.cos(V)) * np.cos(U)
    Y = (c + a * np.cos(V)) * np.sin(U)
    Z = a * np.sin(V)
    return X, Y, Z


if __name__ == '__main__':
    for k in range(3):
        n_points = 100
        coord = torch.rand(n_points, 2)
        _data = torch.rand(0, 2)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                _data = torch.concat([_data, coord + torch.tensor([i, j])])
        _distance_matrix = torch.cdist(_data, _data)
        distance_matrix = torch.zeros(n_points, n_points)
        for i in range(n_points):
            for j in range(n_points):
                distance_matrix[i, j] = torch.min(_distance_matrix[i::n_points, j::n_points])
        assert torch.all(~torch.isnan(distance_matrix))
        assert torch.all(distance_matrix >= 0)

        coord3d = coord.numpy() * 2 * np.pi
        c, a = 4, 2
        data3d = np.asarray([(c + a * np.cos(coord3d[:, 0])) * np.cos(coord3d[:, 1]),
                             (c + a * np.cos(coord3d[:, 0])) * np.sin(coord3d[:, 1]),
                             a * np.sin(coord3d[:, 0])]).T

        x, y, z = plot_torus(50, c, a)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z, antialiased=True, alpha=0.3)
        ax.scatter(*data3d.T, c='C5')
        ax.view_init(elev=70, azim=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'figs/torus_{k}.svg')
        plt.show()
        plt.close()

        inspect_list = np.linspace(0, np.sqrt(2) / 2 / 2, 1000)
        result = Betti_curve(distance_matrix.numpy(), inspect_list, dim=1)

        plt.plot(result)
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f'figs/BettiCurve_{k}.svg')
        plt.show()
        plt.close()
