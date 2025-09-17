import time

import numpy as np


def LEO(population, obj_func, lb, ub, max_iter):
    # Initialize population
    num_agents, num_variables = population.shape[0], population.shape[1]
    fitness = np.array([obj_func(ind) for ind in population])
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    Convergence_curve = np.zeros((max_iter, 1))
    t = 0
    ct = time.time()
    # Main loop
    for t in range(1, max_iter + 1):
        for i in range(num_agents):
            # Phase 1: Teacher Selection and Training
            suggested_teachers = [population[k] for k in range(num_agents) if fitness[k] < fitness[i]] + [best_solution]
            selected_teacher = suggested_teachers[np.random.randint(len(suggested_teachers))]

            r = np.random.rand()
            I = np.random.choice([1, 2])
            new_position = population[i] + r * (selected_teacher - I * population[i])
            new_position = np.clip(new_position, lb, ub)

            new_fitness = obj_func(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

            # Phase 2: Students Learning from Each Other
            other_student = population[np.random.randint(num_agents)]
            if fitness[np.random.randint(num_agents)] < fitness[i]:
                new_position = population[i] + r * (other_student - I * population[i])
            else:
                new_position = population[i] + r * (population[i] - I * other_student)
            new_position = np.clip(new_position, lb, ub)

            new_fitness = obj_func(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

            # Phase 3: Individual Practice
            new_position = population[i] + (lb + r * (ub - lb)) / t
            new_position = np.clip(new_position, lb, ub)

            new_fitness = obj_func(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        # Update the best solution found
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness:
            best_fitness = fitness[current_best_index]
            best_solution = population[current_best_index]

        Convergence_curve[t] = best_fitness
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_solution, ct

