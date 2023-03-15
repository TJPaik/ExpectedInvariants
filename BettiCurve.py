import numpy as np

from dist_mat_gen import sphere_gen, flat_torus_gen
from top_curve import Betti_curve


def sphere(n_points: int, n_dim: int, Betti_dim: int, inspect_list: np.ndarray) -> np.ndarray:
    dist_mat = sphere_gen(n_points, n_dim)
    return Betti_curve(dist_mat, inspect_list, Betti_dim)


def flat_torus(n_points: int, Betti_dim: int, inspect_list: np.ndarray) -> np.ndarray:
    dist_mat = flat_torus_gen(n_points)
    return Betti_curve(dist_mat, inspect_list, Betti_dim)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt

    # Caution: Too many points make it inaccurate.
    sphere_outputs = Parallel(n_jobs=24, verbose=9)(
        delayed(sphere)(32, 1, 1, np.linspace(0, np.pi, 1000)) for _ in range(4000))
    sphere_outputs = np.asarray(sphere_outputs)
    plt.plot(sphere_outputs.mean(0))
    plt.title("Sphere")
    plt.ylim(-1, 2)
    plt.tight_layout()
    plt.show()

    flat_torus_outputs = Parallel(n_jobs=24, verbose=9)(
        delayed(flat_torus)(128, 1, np.linspace(0, np.sqrt(2) / 2, 1000)) for _ in range(4000))
    flat_torus_outputs = np.asarray(flat_torus_outputs)
    plt.plot(flat_torus_outputs.mean(0))
    plt.title("Flat torus")
    plt.tight_layout()
    plt.show()
