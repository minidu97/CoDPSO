import numpy as np
from scipy.stats import wilcoxon


def compute_statistics(results):
    
    #Compute mean, std, best and worst over 30 independent runs and matches the reporting format.
    results = np.array(results)
    return {
        "mean":  float(np.mean(results)),
        "std":   float(np.std(results)),
        "best":  float(np.min(results)),
        "worst": float(np.max(results))
    }


def wilcoxon_test(results_codpso, results_other):
    #Perform Wilcoxon signed-rank test between CoDPSO and a competitor.
    results_codpso = np.array(results_codpso)
    results_other  = np.array(results_other)

    #If results are identical wilcoxon will raise an error
    if np.all(results_codpso == results_other):
        return 0.0, 1.0, "="

    stat, p = wilcoxon(results_codpso, results_other)

    if p < 0.05:
        if np.mean(results_codpso) < np.mean(results_other):
            sign = "+"   #if CoDPSO significantly better
        else:
            sign = "-"   #if CoDPSO significantly worse
    else:
        sign = "="       #if No significant difference

    return stat, p, sign