import numpy as np
from scipy.stats import wilcoxon

def compute_statistics(results):

    return {
        "mean": np.mean(results),
        "std": np.std(results),
        "best": np.min(results),
        "worst": np.max(results)
    }

def wilcoxon_test(results1, results2):

    stat, p = wilcoxon(results1, results2)
    return stat, p