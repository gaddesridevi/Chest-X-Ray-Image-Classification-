import numpy as np
import time



# Mud Ring Algorithm (MRA)
def MRA(population, fobj, VRmin, VRmax, Max_iter):
    num_population, num_dimensions = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    fitness = [fobj(individual) for individual in population]
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        for i in range(num_population):
            # Create a new candidate solution by perturbing the best solution
            candidate_solution = best_solution + 0.1 * np.random.randn(num_dimensions)
            candidate_fitness = fobj(candidate_solution)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            # Replace the current solution if the candidate is better
            if candidate_fitness < fitness[i]:
                population[i] = candidate_solution
                fitness[i] = candidate_fitness

        # Update the best solution found so far
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best_solution = population[best_index]
            best_fitness = fitness[best_index]

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
