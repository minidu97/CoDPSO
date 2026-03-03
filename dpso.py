import numpy as np
from config import SEARCH_BOUNDS

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(
            SEARCH_BOUNDS[0], SEARCH_BOUNDS[1], dim
        )
        self.velocity = np.zeros(dim)

        self.pbest_position = self.position.copy()
        self.pbest_value = np.inf

        self.velocity_history = [self.velocity.copy()]

class Swarm:
    def __init__(self, n_particles, dim):
        self.particles = [Particle(dim) for _ in range(n_particles)]
        self.gbest_position = None
        self.gbest_value = np.inf
        self.stagnancy_counter = 0

    def reinitialize(self, n_particles, dim):
        self.__init__(n_particles, dim)