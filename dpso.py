import numpy as np
import config


class Particle:
    def __init__(self, dim):
        x_min = config.SEARCH_BOUNDS[0]   #-100
        x_max = config.SEARCH_BOUNDS[1]   # 100

        self.position = (
            np.random.rand(dim) * (x_max - x_min) + x_min
        )

        v_max = (x_max - x_min) / 4.0     #= 50
        v_min = -v_max                    #= -50

        self.velocity = (
            np.random.rand(dim) * (v_max - v_min) + v_min
        )

        self.pbest_position = self.position.copy()
        self.pbest_value    = np.inf

        #Velocity history - newest entry first
        self.velocity_history = [self.velocity.copy()]


class Swarm:
    def __init__(self, n_particles, dim):
        self.dim       = dim
        self.particles = [Particle(dim) for _ in range(n_particles)]

        self.gbest_position = self.particles[0].position.copy()
        self.gbest_value    = np.inf

        self.stagnancy_counter = 0   #SC in the paper
        self.n_kill            = 0   #particles removed since last improvement

    #Punitive strategy helpers

    def delete_worst_particle(self):
        #Remove the particle with the highest (worst) pbest value
        if len(self.particles) == 0:
            return
        worst_idx = max(
            range(len(self.particles)),
            key=lambda i: self.particles[i].pbest_value
        )
        self.particles.pop(worst_idx)
        self.n_kill += 1

    def spawn_particle(self):
        #Add a new randomly initialised particle to this swarm
        if len(self.particles) < config.N_MAX:
            self.particles.append(Particle(self.dim))

    def reset_stagnancy_counter(self):
        #Reset SC after a deletion using paper Eq
        self.stagnancy_counter = int(
            config.SC_MAX * (1.0 - 1.0 / (1.0 + self.n_kill))
        )

    @property
    def size(self):
        return len(self.particles)