import numpy as np
import config
from dpso import Swarm, Particle
from complex_operator import compute_memory_velocity


def codpso(problem):
    dim     = config.DIMENSION
    max_fes = config.MAX_FES_FACTOR * dim
    fes     = 0

    #Initialize swarms, positions and velocities
    swarms = [
        Swarm(config.PARTICLES_PER_SWARM, dim)
        for _ in range(config.NUM_SWARMS)
    ]

    #Per-swarm degradation point (Dp) - Stores G_best_old to compare against G_best_new each iteration
    degradation_point   = [None]  * len(swarms)
    gbest_old_per_swarm = [np.inf] * len(swarms)   #track previous gbest

    global_best_value = np.inf

    #Calculate initial fitness values for all particles
    for swarm in swarms:
        for particle in swarm.particles:
            fitness = problem.evaluate(particle.position)
            fes    += 1
            particle.pbest_value    = fitness
            particle.pbest_position = particle.position.copy()
            if fitness < swarm.gbest_value:
                swarm.gbest_value    = fitness
                swarm.gbest_position = particle.position.copy()
            if fitness < global_best_value:
                global_best_value = fitness

    #Main flow
    while fes < max_fes:

        swarms_to_remove = []

        for s_idx, swarm in enumerate(swarms):

            if swarm.size == 0:
                swarms_to_remove.append(s_idx)
                continue

            #Save G_best before this iteration for DE comparison - compare G_best_new vs G_best_old
            gbest_old = gbest_old_per_swarm[s_idx]
            gbest_old_per_swarm[s_idx] = swarm.gbest_value

            #Reset per-iteration kill counter
            swarm.n_kill = 0

            #Calculate velocity vector and Update particle positions
            for particle in swarm.particles:

                #complex-order memory term
                memory = compute_memory_velocity(
                    particle.velocity_history,
                    config.ALPHA,
                    config.BETA,
                    config.MEMORY_LENGTH
                )

                r1 = np.random.rand()
                r2 = np.random.rand()

                cognitive = config.C1 * r1 * (
                    particle.pbest_position - particle.position
                )
                social = config.C2 * r2 * (
                    swarm.gbest_position - particle.position
                )

                new_velocity = memory + cognitive + social

                #clip velocity to -vmax and +vmax
                v_max = (
                    config.SEARCH_BOUNDS[1] - config.SEARCH_BOUNDS[0]
                ) / 4.0
                new_velocity = np.clip(new_velocity, -v_max, v_max)

                #Update position and clip to search bounds
                new_position = particle.position + new_velocity
                new_position = np.clip(
                    new_position,
                    config.SEARCH_BOUNDS[0],
                    config.SEARCH_BOUNDS[1]
                )

                particle.velocity = new_velocity
                particle.position = new_position

                #Update velocity history (newest first, length = r)
                particle.velocity_history.insert(0, new_velocity.copy())
                if len(particle.velocity_history) > config.MEMORY_LENGTH:
                    particle.velocity_history.pop()
            
            #Evaluate fitness of updated particles
            fitness_improved = False

            for particle in swarm.particles:
                fitness = problem.evaluate(particle.position)
                fes    += 1

                #Update personal best
                if fitness < particle.pbest_value:
                    particle.pbest_value    = fitness
                    particle.pbest_position = particle.position.copy()

                #Update swarm best
                if fitness < swarm.gbest_value:
                    swarm.gbest_value    = fitness
                    swarm.gbest_position = particle.position.copy()
                    fitness_improved     = True

                #Update global best
                if fitness < global_best_value:
                    global_best_value = fitness

                if fes >= max_fes:
                    break

            if fitness_improved:
                swarm.spawn_particle()   #adds one particle if N < N_max

            else:
                #Update swarm cycle (increment SC)
                swarm.stagnancy_counter += 1

                #Check if G_best got WORSE than previous
                gbest_degraded = (
                    gbest_old is not None
                    and gbest_old < np.inf
                    and swarm.gbest_value > gbest_old
                )

                if gbest_degraded:
                    #Mark degradation point
                    degradation_point[s_idx] = gbest_old

                    #DE deletion condition
                    if swarm.stagnancy_counter == config.SC_MAX:
                        dp = degradation_point[s_idx]

                        #Find worst particle whose pbest > Dp
                        candidates = [
                            i for i, p in enumerate(swarm.particles)
                            if p.pbest_value > dp
                        ]

                        if candidates:
                            #Delete worst particle among candidates
                            worst_idx = max(
                                candidates,
                                key=lambda i: swarm.particles[i].pbest_value
                            )
                            swarm.particles.pop(worst_idx)
                            swarm.n_kill += 1

                        #Re-initialize SC
                        swarm.reset_stagnancy_counter()

                    #if N < N_min → mark swarm for deletion
                    if swarm.size < config.N_MIN:
                        swarms_to_remove.append(s_idx)
                        continue

                else:
                    #Standard DPSO branch

                    if swarm.stagnancy_counter == config.SC_MAX:

                        swarm.delete_worst_particle()

                        swarm.reset_stagnancy_counter()

                    if swarm.size < config.N_MIN:
                        swarms_to_remove.append(s_idx)
                        continue

            if fes >= max_fes:
                break

        #Remove swarms that fell below N_min
        for idx in sorted(set(swarms_to_remove), reverse=True):
            if idx < len(swarms):
                swarms.pop(idx)
                degradation_point.pop(idx)
                gbest_old_per_swarm.pop(idx)

        #Spawn new swarm
        all_no_kill = all(s.n_kill == 0 for s in swarms)
        if all_no_kill and len(swarms) < config.NS_MAX:
            new_swarm = Swarm(config.PARTICLES_PER_SWARM, dim)
            swarms.append(new_swarm)
            degradation_point.append(None)
            gbest_old_per_swarm.append(np.inf)

        #restart with one fresh swarm
        if len(swarms) == 0:
            swarms              = [Swarm(config.PARTICLES_PER_SWARM, dim)]
            degradation_point   = [None]
            gbest_old_per_swarm = [np.inf]

        #iteration count is tracked implicitly via fes
        if fes >= max_fes:
            break

    return global_best_value