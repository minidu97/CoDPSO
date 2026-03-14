import numpy as np
import config


class Particle:
    def __init__(self, dim):
        x_min = config.SEARCH_BOUNDS[0]
        x_max = config.SEARCH_BOUNDS[1]

        self.position = np.random.rand(dim) * (x_max - x_min) + x_min

        v_max = (x_max - x_min) / 4.0
        v_min = -v_max
        self.velocity = np.random.rand(dim) * (v_max - v_min) + v_min

        self.pbest_position = self.position.copy()
        self.pbest_value    = np.inf

        self.velocity_history = [self.velocity.copy()]


class Swarm:
    def __init__(self, n_particles, dim):
        self.dim       = dim
        self.particles = [Particle(dim) for _ in range(n_particles)]

        self.gbest_position = self.particles[0].position.copy()
        self.gbest_value    = np.inf

        self.stagnancy_counter = 0

        self.n_kill       = 0   #kills this iteration only (resets each iteration)
        self.n_kill_total = 0   #cumulative kills since last improvement — used in Eq.(7)

    def delete_worst_particle(self):
        if len(self.particles) == 0:
            return
        worst_idx = max(
            range(len(self.particles)),
            key=lambda i: self.particles[i].pbest_value
        )
        self.particles.pop(worst_idx)
        self.n_kill       += 1
        self.n_kill_total += 1

    def spawn_particle(self):
        if len(self.particles) < config.N_MAX:
            self.particles.append(Particle(self.dim))

    def reset_stagnancy_counter(self):
        #SC = SC_max * (1 - 1/(1 + N_kill))
        #Uses n_kill_total so SC grows with repeated deletions
        self.stagnancy_counter = int(
            config.SC_MAX * (1.0 - 1.0 / (1.0 + self.n_kill_total))
        )

    def reset_kill_counters(self):
        #called on genuine improvement — restarts the kill cycle
        self.n_kill       = 0
        self.n_kill_total = 0

    @property
    def size(self):
        return len(self.particles)