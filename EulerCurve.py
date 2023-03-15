import numpy as np

from dist_mat_gen import sphere_gen, flat_torus_gen
from top_curve import Euler_curve


def sphere(n_points: int, n_dim: int, inspect_list: np.ndarray) -> np.ndarray:
    dist_mat = sphere_gen(n_points, n_dim)
    return Euler_curve(dist_mat, inspect_list)


def flat_torus(n_points: int, inspect_list: np.ndarray) -> np.ndarray:
    dist_mat = flat_torus_gen(n_points)
    return Euler_curve(dist_mat, inspect_list)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt

    sphere_outputs = Parallel(n_jobs=24, verbose=9)(
        delayed(sphere)(10, 3, np.linspace(0, np.pi, 1000)) for _ in range(4000))
    sphere_outputs = np.asarray(sphere_outputs)
    plt.plot(sphere_outputs.mean(0))
    plt.title("Sphere")
    plt.tight_layout()
    plt.show()

    flat_torus_outputs = Parallel(n_jobs=24, verbose=9)(
        delayed(flat_torus)(10, np.linspace(0, np.sqrt(2) / 2, 1000)) for _ in range(4000))
    flat_torus_outputs = np.asarray(flat_torus_outputs)
    plt.plot(flat_torus_outputs.mean(0))
    plt.title("Flat torus")
    plt.tight_layout()
    plt.show()
