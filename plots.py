import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ----------------------------------------------------------------
# Output folder for all saved figures
# ----------------------------------------------------------------
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Shared style settings (matches paper style)
# ----------------------------------------------------------------
ALGO_COLORS = {
    "CoDPSO":  "black",
    "FDPSO":   "red",
    "TAPSO":   "blue",
    "AWPSO":   "green",
    "XPSO":    "cyan",
    "CoPSO":   "magenta",
    "BFL-PSO": "orange",
    "APD-PSO": "purple",
    "EOPSO":   "brown",
    "HPBPSO":  "pink",
    "CoQPSO":  "gray",
}

ALGO_LINESTYLES = {
    "CoDPSO":  "-",
    "FDPSO":   "-",
    "TAPSO":   "-",
    "AWPSO":   "-",
    "XPSO":    "-",
    "CoPSO":   "--",
    "BFL-PSO": "--",
    "APD-PSO": "--",
    "EOPSO":   "--",
    "HPBPSO":  "--",
    "CoQPSO":  "--",
}

FUNC_NAMES = {
    1:  "F1 - Zakharov",
    2:  "F2 - Rosenbrock",
    3:  "F3 - Schaffer f6",
    4:  "F4 - Rastrigin",
    5:  "F5 - Levy",
    6:  "F6 - Hybrid 1",
    7:  "F7 - Hybrid 2",
    8:  "F8 - Hybrid 3",
    9:  "F9 - Composition 1",
    10: "F10 - Composition 2",
    11: "F11 - Composition 3",
    12: "F12 - Composition 4",
}

# Representative functions used in paper Figs 8,9,10,11,12,13
REPRESENTATIVE_FIDS = [1, 4, 7, 11]


# ================================================================
# PLOT 1 — Convergence Curves  (paper Figs 8 & 9)
# ================================================================

def plot_convergence_curves(convergence_data, dim, fids=None):
    """
    Plot convergence curves: best fitness value vs NFE.

    Parameters
    ----------
    convergence_data : dict
        {algo_name: {fid: np.ndarray of shape (runs, nfe_steps)}}
        Each row is one run, each column is one NFE checkpoint.
    dim : int
        Problem dimension (10 or 20), used for title and filename.
    fids : list of int, optional
        Function IDs to plot. Defaults to REPRESENTATIVE_FIDS.
    """
    if fids is None:
        fids = REPRESENTATIVE_FIDS

    n_funcs = len(fids)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    max_fes = 1000 * dim

    for ax_idx, fid in enumerate(fids):
        ax = axes[ax_idx]

        for algo, fid_data in convergence_data.items():
            if fid not in fid_data:
                continue

            data = fid_data[fid]          # shape: (runs, nfe_steps)
            mean_curve = np.mean(data, axis=0)
            nfe_axis   = np.linspace(0, max_fes, len(mean_curve))

            ax.semilogy(
                nfe_axis,
                mean_curve,
                label=algo,
                color=ALGO_COLORS.get(algo, "black"),
                linestyle=ALGO_LINESTYLES.get(algo, "-"),
                linewidth=1.5
            )

        ax.set_title(FUNC_NAMES.get(fid, f"F{fid}"), fontsize=11)
        ax.set_xlabel("NFE", fontsize=10)
        ax.set_ylabel("Benchmark Function Value", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Convergence Performance — CEC 2022 (D={dim})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"convergence_D{dim}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ================================================================
# PLOT 2 — Box Plots  (paper Figs 10 & 11)
# ================================================================

def plot_boxplots(results_data, dim, fids=None):
    """
    Plot box plots of 30-run results per algorithm per function.

    Parameters
    ----------
    results_data : dict
        {algo_name: {fid: np.ndarray of shape (30,)}}
    dim : int
        Problem dimension (10 or 20).
    fids : list of int, optional
        Function IDs to plot. Defaults to REPRESENTATIVE_FIDS.
    """
    if fids is None:
        fids = REPRESENTATIVE_FIDS

    n_funcs = len(fids)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    algo_names = list(results_data.keys())

    for ax_idx, fid in enumerate(fids):
        ax = axes[ax_idx]

        box_data = []
        for algo in algo_names:
            if fid in results_data[algo]:
                box_data.append(results_data[algo][fid])
            else:
                box_data.append(np.array([np.nan]))

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            notch=False,
            showfliers=True,
            flierprops=dict(marker="o", markersize=4, linestyle="none")
        )

        # Colour the CoDPSO box differently to highlight it
        for patch_idx, patch in enumerate(bp["boxes"]):
            if algo_names[patch_idx] == "CoDPSO":
                patch.set_facecolor("lightblue")
            else:
                patch.set_facecolor("white")

        ax.set_title(FUNC_NAMES.get(fid, f"F{fid}"), fontsize=11)
        ax.set_xticks(range(1, len(algo_names) + 1))
        ax.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Error", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Box Plots — CEC 2022 (D={dim})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"boxplots_D{dim}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ================================================================
# PLOT 3 — Bar Charts with Error Bars  (paper Figs 12 & 13)
# ================================================================

def plot_bar_charts(results_data, dim, fids=None):
    """
    Plot average fitness ± std deviation as bar charts with error bars.

    Parameters
    ----------
    results_data : dict
        {algo_name: {fid: np.ndarray of shape (30,)}}
    dim : int
        Problem dimension (10 or 20).
    fids : list of int, optional
        Function IDs to plot. Defaults to REPRESENTATIVE_FIDS.
    """
    if fids is None:
        fids = REPRESENTATIVE_FIDS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    algo_names = list(results_data.keys())
    x_pos      = np.arange(len(algo_names))

    for ax_idx, fid in enumerate(fids):
        ax = axes[ax_idx]

        means = []
        stds  = []
        for algo in algo_names:
            if fid in results_data[algo]:
                data = results_data[algo][fid]
                means.append(np.mean(data))
                stds.append(np.std(data))
            else:
                means.append(np.nan)
                stds.append(0)

        # Highlight CoDPSO bar in red (matches paper Fig 12/13 style)
        bar_colors = [
            "red" if a == "CoDPSO" else "steelblue"
            for a in algo_names
        ]

        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            color=bar_colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
            error_kw=dict(elinewidth=1, ecolor="black")
        )

        ax.set_title(
            f"Performance Comparison ({FUNC_NAMES.get(fid, f'F{fid}')}-{dim}D) "
            f"- Avg. ± Std Dev",
            fontsize=9
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Average Fitness", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Average Fitness ± Std Dev — CEC 2022 (D={dim})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"bar_charts_D{dim}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ================================================================
# PLOT 4 — Sensitivity Analysis Scatter  (paper Figs 3 & 4)
# ================================================================

def plot_sensitivity_scatter(rank_data, dim):
    """
    Plot the (alpha, beta) locus of best complex-order area.
    Matches paper Figs 3 & 4 — circle markers at the top-ranked
    (alpha, beta) combinations across all 12 benchmark functions.

    Parameters
    ----------
    rank_data : dict
        {(alpha, beta): total_rank}
        Total rank = sum of ranks across all 12 functions (lower = better).
        Only the TOP 10 combinations are plotted (paper Tables 10 & 11).
    dim : int
        Problem dimension (10 or 20), used for title and filename.
    """
    # Sort by total rank ascending (best first)
    sorted_pairs = sorted(rank_data.items(), key=lambda x: x[1])
    top_10       = sorted_pairs[:10]

    alphas = [p[0][0] for p in top_10]
    betas  = [p[0][1] for p in top_10]
    ranks  = [p[1]    for p in top_10]

    # Best point (rank 1)
    best_alpha, best_beta = top_10[0][0]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot all top-10 combinations as circles
    scatter = ax.scatter(
        alphas, betas,
        s=120,
        c="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
        label="Top 10 (α, β)"
    )

    # Highlight the optimal point
    ax.scatter(
        [best_alpha], [best_beta],
        s=200,
        c="white",
        edgecolors="black",
        linewidths=2.5,
        zorder=4,
        marker="o"
    )

    # Annotate the optimal point with X and Y labels (matches paper style)
    ax.annotate(
        f"X {best_alpha}\nY {best_beta}",
        xy=(best_alpha, best_beta),
        xytext=(best_alpha + 0.05, best_beta - 0.08),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )

    # Add "Order" label box in top-right corner (matches paper)
    ax.text(
        0.97, 0.97, "Order",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
    )

    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("β", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_title(
        f"Best Complex-Order Area — {dim}-dimensional",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"sensitivity_scatter_D{dim}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ================================================================
# Helper — record convergence history inside codpso run
# ================================================================

def record_convergence(problem, dim, alpha, beta, memory_length,
                       c1, c2, search_bounds, max_fes_factor,
                       num_swarms, particles_per_swarm, sc_max,
                       ns_max, n_min, n_max):
    """
    Runs ONE CoDPSO trial and returns the convergence curve
    (best fitness at every function evaluation).

    Returns
    -------
    np.ndarray of shape (max_fes,) — best fitness at each FE step.
    """
    from dpso import Swarm
    from complex_operator import compute_memory_velocity

    max_fes   = max_fes_factor * dim
    fes       = 0
    curve     = np.full(max_fes, np.inf)

    swarms              = [Swarm(particles_per_swarm, dim) for _ in range(num_swarms)]
    degradation_point   = [None]  * len(swarms)
    gbest_old_per_swarm = [np.inf] * len(swarms)
    global_best_value   = np.inf

    # Initial evaluation
    for swarm in swarms:
        for particle in swarm.particles:
            fitness = problem.evaluate(particle.position)
            if fes < max_fes:
                if fitness < particle.pbest_value:
                    particle.pbest_value    = fitness
                    particle.pbest_position = particle.position.copy()
                if fitness < swarm.gbest_value:
                    swarm.gbest_value    = fitness
                    swarm.gbest_position = particle.position.copy()
                if fitness < global_best_value:
                    global_best_value = fitness
                curve[fes] = global_best_value
                fes += 1

    while fes < max_fes:
        swarms_to_remove = []

        for s_idx, swarm in enumerate(swarms):
            if swarm.size == 0:
                swarms_to_remove.append(s_idx)
                continue

            gbest_old = gbest_old_per_swarm[s_idx]
            gbest_old_per_swarm[s_idx] = swarm.gbest_value
            swarm.n_kill = 0

            v_max = (search_bounds[1] - search_bounds[0]) / 4.0

            for particle in swarm.particles:
                memory = compute_memory_velocity(
                    particle.velocity_history, alpha, beta, memory_length
                )
                r1 = np.random.rand()
                r2 = np.random.rand()
                cognitive    = c1 * r1 * (particle.pbest_position - particle.position)
                social       = c2 * r2 * (swarm.gbest_position    - particle.position)
                new_velocity = np.clip(memory + cognitive + social, -v_max, v_max)
                new_position = np.clip(
                    particle.position + new_velocity,
                    search_bounds[0], search_bounds[1]
                )
                particle.velocity = new_velocity
                particle.position = new_position
                particle.velocity_history.insert(0, new_velocity.copy())
                if len(particle.velocity_history) > memory_length:
                    particle.velocity_history.pop()

            fitness_improved = False
            for particle in swarm.particles:
                fitness = problem.evaluate(particle.position)
                if fes < max_fes:
                    if fitness < particle.pbest_value:
                        particle.pbest_value    = fitness
                        particle.pbest_position = particle.position.copy()
                    if fitness < swarm.gbest_value:
                        swarm.gbest_value    = fitness
                        swarm.gbest_position = particle.position.copy()
                        fitness_improved     = True
                    if fitness < global_best_value:
                        global_best_value = fitness
                    curve[fes] = global_best_value
                    fes += 1
                if fes >= max_fes:
                    break

            if fitness_improved:
                swarm.spawn_particle()
            else:
                swarm.stagnancy_counter += 1
                gbest_degraded = (
                    gbest_old is not None and gbest_old < np.inf
                    and swarm.gbest_value > gbest_old
                )
                if gbest_degraded:
                    degradation_point[s_idx] = gbest_old
                    if swarm.stagnancy_counter == sc_max:
                        dp         = degradation_point[s_idx]
                        candidates = [
                            i for i, p in enumerate(swarm.particles)
                            if p.pbest_value > dp
                        ]
                        if candidates:
                            worst_idx = max(candidates,
                                            key=lambda i: swarm.particles[i].pbest_value)
                            swarm.particles.pop(worst_idx)
                            swarm.n_kill += 1
                        swarm.reset_stagnancy_counter()
                    if swarm.size < n_min:
                        swarms_to_remove.append(s_idx)
                        continue
                else:
                    if swarm.stagnancy_counter == sc_max:
                        swarm.delete_worst_particle()
                        swarm.reset_stagnancy_counter()
                    if swarm.size < n_min:
                        swarms_to_remove.append(s_idx)
                        continue

            if fes >= max_fes:
                break

        for idx in sorted(set(swarms_to_remove), reverse=True):
            if idx < len(swarms):
                swarms.pop(idx)
                degradation_point.pop(idx)
                gbest_old_per_swarm.pop(idx)

        all_no_kill = all(s.n_kill == 0 for s in swarms)
        if all_no_kill and len(swarms) < ns_max:
            swarms.append(Swarm(particles_per_swarm, dim))
            degradation_point.append(None)
            gbest_old_per_swarm.append(np.inf)

        if len(swarms) == 0:
            swarms              = [Swarm(particles_per_swarm, dim)]
            degradation_point   = [None]
            gbest_old_per_swarm = [np.inf]

        if fes >= max_fes:
            break

    # Forward-fill any unfilled entries
    for i in range(1, max_fes):
        if curve[i] == np.inf:
            curve[i] = curve[i - 1]

    return curve


# ================================================================
# Example usage — called from main.py or standalone
# ================================================================

if __name__ == "__main__":
    import config
    from cec_wrapper import get_cec2022_function

    print("Generating sensitivity scatter plots using paper Table 10 & 11 data...")

    # --- Fig 3: D=10 top-10 (alpha, beta) from paper Table 10 ---
    rank_data_10d = {
        (0.3, 0.8): 332,
        (0.2, 0.8): 335,
        (0.7, 0.5): 350,
        (0.6, 0.1): 374,
        (0.6, 0.4): 378,
        (0.7, 0.0): 400,
        (0.5, 0.6): 402,
        (0.7, 0.4): 411,
        (0.3, 0.9): 426,
        (0.7, 0.2): 429,
    }
    plot_sensitivity_scatter(rank_data_10d, dim=10)

    # --- Fig 4: D=20 top-10 (alpha, beta) from paper Table 11 ---
    rank_data_20d = {
        (0.8, 0.0): 373,
        (0.1, 0.7): 389,
        (0.8, 0.1): 408,
        (0.2, 0.9): 417,
        (0.0, 0.7): 451,
        (0.1, 0.8): 467,
        (0.7, 0.5): 472,
        (0.6, 0.6): 482,
        (0.7, 0.1): 491,
        (0.1, 0.6): 497,
    }
    plot_sensitivity_scatter(rank_data_20d, dim=20)

    print("\nRunning convergence, box plot and bar chart for F1 D=10 as demo...")

    fid     = 1
    dim     = config.DIMENSION
    problem = get_cec2022_function(fid, dim)
    runs    = 5    # use 5 for quick demo; set to 30 for full paper results

    all_curves  = []
    all_results = []

    for run in range(runs):
        print(f"  Run {run + 1}/{runs}", end="\r")
        curve = record_convergence(
            problem         = problem,
            dim             = dim,
            alpha           = config.ALPHA,
            beta            = config.BETA,
            memory_length   = config.MEMORY_LENGTH,
            c1              = config.C1,
            c2              = config.C2,
            search_bounds   = config.SEARCH_BOUNDS,
            max_fes_factor  = config.MAX_FES_FACTOR,
            num_swarms      = config.NUM_SWARMS,
            particles_per_swarm = config.PARTICLES_PER_SWARM,
            sc_max          = config.SC_MAX,
            ns_max          = config.NS_MAX,
            n_min           = config.N_MIN,
            n_max           = config.N_MAX,
        )
        all_curves.append(curve)
        all_results.append(curve[-1])

    convergence_data = {"CoDPSO": {fid: np.array(all_curves)}}
    results_data     = {"CoDPSO": {fid: np.array(all_results)}}

    plot_convergence_curves(convergence_data, dim=dim, fids=[fid])
    plot_boxplots(results_data,    dim=dim, fids=[fid])
    plot_bar_charts(results_data,  dim=dim, fids=[fid])