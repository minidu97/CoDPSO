import numpy as np
import config
from codpso import codpso
from cec_wrapper import get_cec2022_function
from statistics import compute_statistics


def run_experiment(fid):

    problem = get_cec2022_function(fid, config.DIMENSION)

    results = []

    for run in range(config.RUNS):
        print(f"Function {fid} | Run {run+1}")
        result = codpso(problem)
        results.append(result)

    stats = compute_statistics(results)

    print("\nFinal Statistics:")
    print(stats)

    return results


if __name__ == "__main__":

    for fid in range(1, 13):
        print(f"\n===== Running CEC 2022 F{fid} =====")
        run_experiment(fid)