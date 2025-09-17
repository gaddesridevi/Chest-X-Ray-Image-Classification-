import random
import numpy as np
import time


# Golden Eagle Optimizer (GEO)
def GEO(agents, fobj, VRmin, VRmax, Max_iter):
    num_agents, dim = agents.shape[0], agents.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    verbose = True

    best_agent = np.zeros((1, dim))
    best_fitness = float('inf')

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    # Main loop
    for t in range(Max_iter):
        # Calculate fitness for all agents
        fitness_values = [fobj(agent) for agent in agents]

        # Sort agents based on their fitness (ascending order)
        sorted_agents = [x for _, x in sorted(zip(fitness_values, agents))]
        agents = sorted_agents

        # Golden Eagle's update position
        c1 = 1.5  # Dive speed
        c2 = 1.5  # Search speed

        for i in range(num_agents):
            rand = random.random()
            if rand < 0.5:
                agents[i] = agents[i] - c1 * rand * (agents[i] - agents[0])
            else:
                agents[i] = agents[i] + c2 * rand * (agents[-1] - agents[i])

            # Ensure agents' positions are within the search space
            agents[i] = np.clip(agents[i], lb, ub)

        # Print the best fitness value in the current iteration
        if verbose:
            best_fitness = fitness_values[0]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_agent, ct
