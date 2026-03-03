import numpy as np
from dpso import Swarm
from complex_operator import compute_memory_velocity
import config


def codpso(problem):

    dim = config.DIMENSION
    max_fes = config.MAX_FES_FACTOR * dim
    fes = 0

    swarms = [
        Swarm(config.PARTICLES_PER_SWARM, dim)
        for _ in range(config.NUM_SWARMS)
    ]

    global_best = np.inf

    while fes < max_fes:

        for swarm in swarms:

            improved = False

            # Evaluate particles
            for particle in swarm.particles:

                fitness = problem.evaluate(particle.position)
                fes += 1

                if fitness < particle.pbest_value:
                    particle.pbest_value = fitness
                    particle.pbest_position = particle.position.copy()

                if fitness < swarm.gbest_value:
                    swarm.gbest_value = fitness
                    swarm.gbest_position = particle.position.copy()
                    improved = True

                if fitness < global_best:
                    global_best = fitness

                if fes >= max_fes:
                    break

            # Stagnation handling
            if not improved:
                swarm.stagnancy_counter += 1
            else:
                swarm.stagnancy_counter = 0

            if swarm.stagnancy_counter > config.SC_MAX:
                swarm.reinitialize(config.PARTICLES_PER_SWARM, dim)
                continue

            # Update velocities and positions
            for particle in swarm.particles:

                memory = compute_memory_velocity(
                    particle.velocity_history,
                    config.ALPHA,
                    config.BETA,
                    config.MEMORY_LENGTH
                )

                r1, r2 = np.random.rand(), np.random.rand()

                cognitive = config.C1 * r1 * (
                    particle.pbest_position - particle.position
                )
                social = config.C2 * r2 * (
                    swarm.gbest_position - particle.position
                )

                new_velocity = memory + cognitive + social

                particle.velocity = new_velocity
                particle.position += new_velocity

                particle.position = np.clip(
                    particle.position,
                    config.SEARCH_BOUNDS[0],
                    config.SEARCH_BOUNDS[1]
                )

                particle.velocity_history.insert(0, new_velocity.copy())

                if len(particle.velocity_history) > config.MEMORY_LENGTH:
                    particle.velocity_history.pop()

            if fes >= max_fes:
                break

    return global_best