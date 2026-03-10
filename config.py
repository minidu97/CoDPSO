DIMENSION = 10                  
SEARCH_BOUNDS = (-100, 100)
MAX_FES_FACTOR = 1000           #Max FEs = 1000 * D

#based on table 14
ALPHA = 0.3                     #Real part of complex order
BETA  = 0.8                     #Imaginary part of complex order
MEMORY_LENGTH = 4               #r=4

W  = 1.0                        #Inertia weight
C1 = 1.5                        #Cognitive coefficient
C2 = 1.5                        #Social coefficient

NUM_SWARMS     = 5              #Initial number of swarms
NS_MIN         = 3              #Minimum number of swarms
NS_MAX         = 10             #Maximum number of swarms

PARTICLES_PER_SWARM = 10        #Initial particles per swarm
N_MIN          = 3             #Minimum particles per swarm
N_MAX          = 50             #Maximum particles per swarm

SC_MAX         = 30             #Stagnancy counter limit

RUNS = 30                       #Independent runs per function