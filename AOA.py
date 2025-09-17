import numpy as np
import time


# Archimedes Optimization Algorithm (AOA)
def AOA(points, fobj, VRmin, VRmax, Max_iter):
    num_points, dim = points.shape[0], points.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    best_solution = None
    best_fitness = float('inf')

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        fitness = [fobj(point) for point in points]

        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best_solution = points[best_index]
            best_fitness = fitness[best_index]

        # Estimate the optimal solution using the ratio of points inside a hypercube
        feasible_count = np.sum(np.all((points >= 0) & (points <= 1), axis=1))
        if feasible_count > 0:
            estimated_optimal = feasible_count / num_points
        feasible_count= np.clip(feasible_count, lb, ub)
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct

