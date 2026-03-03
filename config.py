# ===============================
# Problem Settings
# ===============================
DIMENSION = 10
SEARCH_BOUNDS = (-100, 100)
MAX_FES_FACTOR = 10000   # Max FEs = 10000 * D

# ===============================
# CoDPSO Parameters
# ===============================
ALPHA = 0.2
BETA = 0.8
MEMORY_LENGTH = 4

W = 1.0
C1 = 1.5
C2 = 1.5

# ===============================
# DPSO Parameters
# ===============================
NUM_SWARMS = 5
PARTICLES_PER_SWARM = 30
SC_MAX = 30   # stagnation limit

# ===============================
# Experiment Settings
# ===============================
RUNS = 30