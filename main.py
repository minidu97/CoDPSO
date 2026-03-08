import numpy as np
import config
from codpso import codpso
from cec_wrapper import get_cec2022_function
from stats_utils import compute_statistics
from plots import (plot_convergence_curves, plot_boxplots,
                   plot_bar_charts, plot_sensitivity_scatter,
                   record_convergence)


def run_experiment(fid):
    problem = get_cec2022_function(fid, config.DIMENSION)
    results = []
    curves  = []

    for run in range(config.RUNS):
        print(f"  Function F{fid} | Run {run + 1:02d}/{config.RUNS}", end="\r")
        best_val = codpso(problem)
        results.append(best_val)

        #record convergence curve for this run
        curve = record_convergence(
            problem             = problem,
            dim                 = config.DIMENSION,
            alpha               = config.ALPHA,
            beta                = config.BETA,
            memory_length       = config.MEMORY_LENGTH,
            c1                  = config.C1,
            c2                  = config.C2,
            search_bounds       = config.SEARCH_BOUNDS,
            max_fes_factor      = config.MAX_FES_FACTOR,
            num_swarms          = config.NUM_SWARMS,
            particles_per_swarm = config.PARTICLES_PER_SWARM,
            sc_max              = config.SC_MAX,
            ns_max              = config.NS_MAX,
            n_min               = config.N_MIN,
            n_max               = config.N_MAX,
        )
        curves.append(curve)

    stats = compute_statistics(results)

    print(f"\n  F{fid} Results over {config.RUNS} runs:")
    print(f"    Mean  : {stats['mean']:.4e}")
    print(f"    Std   : {stats['std']:.4e}")
    print(f"    Best  : {stats['best']:.4e}")
    print(f"    Worst : {stats['worst']:.4e}")

    return results, np.array(curves)


if __name__ == "__main__":
    print(f"CoDPSO | Dimension={config.DIMENSION} | "
          f"alpha={config.ALPHA} | beta={config.BETA} | "
          f"r={config.MEMORY_LENGTH}")
    print("=" * 55)

    all_results = {}
    all_curves  = {}

    for fid in range(1, 13):
        print(f"\n===== CEC 2022 F{fid} =====")
        results, curves  = run_experiment(fid)
        all_results[fid] = results
        all_curves[fid]  = curves

    print("\n===== All experiments complete =====")

    #generate all plots
    dim = config.DIMENSION

    convergence_data = {"CoDPSO": all_curves}
    results_data     = {"CoDPSO": {fid: np.array(r) for fid, r in all_results.items()}}

    print("\nGenerating plots...")

    #Convergence curves
    plot_convergence_curves(convergence_data, dim=dim, fids=[1, 4, 7, 11])

    #Box plots
    plot_boxplots(results_data, dim=dim, fids=[1, 4, 7, 11])

    #Bar charts with error bars
    plot_bar_charts(results_data, dim=dim, fids=[1, 4, 7, 11])

    #Sensitivity scatter (paper Table 10 & 11 data)
    rank_data_10d = {
        (0.3, 0.8): 332, (0.2, 0.8): 335, (0.7, 0.5): 350,
        (0.6, 0.1): 374, (0.6, 0.4): 378, (0.7, 0.0): 400,
        (0.5, 0.6): 402, (0.7, 0.4): 411, (0.3, 0.9): 426,
        (0.7, 0.2): 429,
    }
    rank_data_20d = {
        (0.8, 0.0): 373, (0.1, 0.7): 389, (0.8, 0.1): 408,
        (0.2, 0.9): 417, (0.0, 0.7): 451, (0.1, 0.8): 467,
        (0.7, 0.5): 472, (0.6, 0.6): 482, (0.7, 0.1): 491,
        (0.1, 0.6): 497,
    }
    plot_sensitivity_scatter(rank_data_10d, dim=10)
    plot_sensitivity_scatter(rank_data_20d, dim=20)

    print("\nAll plots saved to figures/ folder.")