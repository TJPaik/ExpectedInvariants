import numpy as np
from gudhi.rips_complex import RipsComplex
from ripser import ripser


def Euler_curve(distance_matrix: np.ndarray, inspect_list: np.ndarray):
    rips = RipsComplex(distance_matrix=distance_matrix)
    tree = rips.create_simplex_tree(max_dimension=len(distance_matrix) + 1)

    euler_characteristic_list = []
    generator = tree.get_filtration()

    tmp = 0
    f = -np.inf
    for inspect_number in inspect_list:
        while True:
            if f <= inspect_number:
                try:
                    s, f = next(generator)
                    # print(f)
                    s = 1 if len(s) % 2 else -1
                    tmp += s
                except StopIteration:
                    s = 0
                    f = np.inf
            else:
                euler_characteristic_list.append(tmp - s)
                break
    return euler_characteristic_list


def Betti_curve(distance_matrix: np.ndarray, inspect_list: np.ndarray, dim=1, coeff=2):
    diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=dim, coeff=coeff)['dgms'][dim]
    result = np.zeros(len(inspect_list))
    _result = np.digitize(diagrams, inspect_list)
    for el in _result:
        result[el[0]: el[1]] += 1
    return result
