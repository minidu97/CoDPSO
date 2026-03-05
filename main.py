import numpy as np
import config
from codpso import codpso
from cec_wrapper import get_cec2022_function
from stats_utils import compute_statistics   #fixed: was 'statistics' (built-in conflict)


def run_experiment(fid):
    """
    Run CoDPSO on CEC2022 function fid for config.RUNS independent runs.
    Prints mean, std, best, worst matching paper Table format.
    """
    problem = get_cec2022_function(fid, config.DIMENSION)
    results = []

    for run in range(config.RUNS):
        print(f"  Function F{fid} | Run {run + 1:02d}/{config.RUNS}", end="\r")
        best_val = codpso(problem)
        results.append(best_val)

    stats = compute_statistics(results)

    print(f"\n  F{fid} Results over {config.RUNS} runs:")
    print(f"    Mean  : {stats['mean']:.4e}")
    print(f"    Std   : {stats['std']:.4e}")
    print(f"    Best  : {stats['best']:.4e}")
    print(f"    Worst : {stats['worst']:.4e}")

    return results


if __name__ == "__main__":
    print(f"CoDPSO | Dimension={config.DIMENSION} | "
          f"alpha={config.ALPHA} | beta={config.BETA} | "
          f"r={config.MEMORY_LENGTH}")
    print("=" * 55)

    all_results = {}
    for fid in range(1, 13):
        print(f"\n===== CEC 2022 F{fid} =====")
        all_results[fid] = run_experiment(fid)

    print("\n===== All experiments complete =====")