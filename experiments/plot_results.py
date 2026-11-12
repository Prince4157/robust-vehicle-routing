#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality plots from experiment CSV results.

Reads CSV files from the results/ directory and produces PNG figures.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

# Consistent colour palette
ALGO_COLORS = {
    "NN":              "#1f77b4",
    "NN+2opt":         "#ff7f0e",
    "Nominal MILP":    "#2ca02c",
    "Robust MILP":     "#d62728",
    "BS-Exact":        "#9467bd",
    "BS-Heuristic":    "#8c564b",
    "Robust-2opt":     "#e377c2",
    "Robust-NN-2opt":  "#17becf",
}

ALGO_MARKERS = {
    "NN":              "o",
    "NN+2opt":         "s",
    "Nominal MILP":    "^",
    "Robust MILP":     "D",
    "BS-Exact":        "v",
    "BS-Heuristic":    "P",
    "Robust-2opt":     "*",
    "Robust-NN-2opt":  "X",
}


def _color(alg):
    return ALGO_COLORS.get(alg, "#333333")


def _marker(alg):
    return ALGO_MARKERS.get(alg, "o")


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Small Instance Comparison (bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_small_comparison():
    csv_path = os.path.join(RESULTS_DIR, "exp1_small.csv")
    if not os.path.exists(csv_path):
        print("  ✗ exp1_small.csv not found — skipping")
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["worst_case_cost"])

    sizes = sorted(df["n"].unique())
    algorithms = df["algorithm"].unique()
    n_alg = len(algorithms)
    n_sizes = len(sizes)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_alg
    x = np.arange(n_sizes)

    for i, alg in enumerate(algorithms):
        alg_df = df[df["algorithm"] == alg]
        vals = []
        for s in sizes:
            subset = alg_df[alg_df["n"] == s]
            vals.append(subset["worst_case_cost"].values[0] if len(subset) > 0 else 0)
        offset = (i - n_alg / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=alg, color=_color(alg),
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Instance Size (n)", fontsize=12)
    ax.set_ylabel("Worst-Case Cost", fontsize=12)
    ax.set_title("Algorithm Comparison on Small Instances", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig_small_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Scalability (runtime vs n, log scale)
# ═══════════════════════════════════════════════════════════════════════════

def plot_scalability():
    csv_path = os.path.join(RESULTS_DIR, "exp2_scalability.csv")
    if not os.path.exists(csv_path):
        print("  ✗ exp2_scalability.csv not found — skipping")
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["runtime"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for alg in df["algorithm"].unique():
        alg_df = df[df["algorithm"] == alg].sort_values("n")
        ax.plot(alg_df["n"], alg_df["runtime"],
                marker=_marker(alg), color=_color(alg),
                label=alg, linewidth=2, markersize=8)

    ax.set_xlabel("Instance Size (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=12)
    ax.set_title("Algorithm Scalability", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, "fig_scalability.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Γ Sensitivity (two subplots)
# ═══════════════════════════════════════════════════════════════════════════

def plot_gamma_sensitivity():
    csv_path = os.path.join(RESULTS_DIR, "exp3_gamma_sensitivity.csv")
    if not os.path.exists(csv_path):
        print("  ✗ exp3_gamma_sensitivity.csv not found — skipping")
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["worst_case_cost"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    algorithms = df["algorithm"].unique()

    # --- (a) Worst-case cost vs Γ ---
    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg].sort_values("gamma")
        ax1.plot(alg_df["gamma"], alg_df["worst_case_cost"],
                 marker=_marker(alg), color=_color(alg),
                 label=alg, linewidth=2, markersize=7)
    ax1.set_xlabel("Γ (Budget of Uncertainty)", fontsize=11)
    ax1.set_ylabel("Worst-Case Cost", fontsize=11)
    ax1.set_title("(a) Worst-Case Cost vs Γ", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- (b) Price of Robustness vs Γ ---
    # Use NN+2opt at Γ=0 as baseline nominal cost for each algorithm
    nn2opt_g0 = df[(df["algorithm"] == "NN+2opt") & (df["gamma"] == 0)]
    if len(nn2opt_g0) > 0:
        baseline_wc = nn2opt_g0["worst_case_cost"].values[0]
    else:
        # Fallback: use NN at Γ=0
        nn_g0 = df[(df["algorithm"] == "NN") & (df["gamma"] == 0)]
        baseline_wc = nn_g0["worst_case_cost"].values[0] if len(nn_g0) > 0 else 1.0

    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg].sort_values("gamma")
        por = (alg_df["worst_case_cost"].values - baseline_wc) / baseline_wc
        ax2.plot(alg_df["gamma"].values, por,
                 marker=_marker(alg), color=_color(alg),
                 label=alg, linewidth=2, markersize=7)
    ax2.set_xlabel("Γ (Budget of Uncertainty)", fontsize=11)
    ax2.set_ylabel("Price of Robustness", fontsize=11)
    ax2.set_title("(b) Price of Robustness vs Γ", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Sensitivity Analysis: Budget of Uncertainty (Γ)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_gamma_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: ξ Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def plot_xi_sensitivity():
    csv_path = os.path.join(RESULTS_DIR, "exp4_xi_sensitivity.csv")
    if not os.path.exists(csv_path):
        print("  ✗ exp4_xi_sensitivity.csv not found — skipping")
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["worst_case_cost"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    algorithms = df["algorithm"].unique()

    # --- (a) Worst-case cost vs ξ ---
    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg].sort_values("xi")
        ax1.plot(alg_df["xi"], alg_df["worst_case_cost"],
                 marker=_marker(alg), color=_color(alg),
                 label=alg, linewidth=2, markersize=7)
    ax1.set_xlabel("ξ (Uncertainty Level)", fontsize=11)
    ax1.set_ylabel("Worst-Case Cost", fontsize=11)
    ax1.set_title("(a) Worst-Case Cost vs ξ", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- (b) Price of Robustness vs ξ ---
    # Baseline: NN+2opt at ξ=0
    nn2opt_x0 = df[(df["algorithm"] == "NN+2opt") & (df["xi"] == 0.0)]
    if len(nn2opt_x0) > 0:
        baseline_wc = nn2opt_x0["worst_case_cost"].values[0]
    else:
        nn_x0 = df[(df["algorithm"] == "NN") & (df["xi"] == 0.0)]
        baseline_wc = nn_x0["worst_case_cost"].values[0] if len(nn_x0) > 0 else 1.0

    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg].sort_values("xi")
        por = (alg_df["worst_case_cost"].values - baseline_wc) / baseline_wc
        ax2.plot(alg_df["xi"].values, por,
                 marker=_marker(alg), color=_color(alg),
                 label=alg, linewidth=2, markersize=7)
    ax2.set_xlabel("ξ (Uncertainty Level)", fontsize=11)
    ax2.set_ylabel("Price of Robustness", fontsize=11)
    ax2.set_title("(b) Price of Robustness vs ξ", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Sensitivity Analysis: Uncertainty Level (ξ)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_xi_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Monte Carlo Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def plot_monte_carlo():
    csv_path = os.path.join(RESULTS_DIR, "exp5_monte_carlo.csv")
    if not os.path.exists(csv_path):
        print("  ✗ exp5_monte_carlo.csv not found — skipping")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("  ✗ exp5_monte_carlo.csv is empty — skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = df["algorithm"].values
    x = np.arange(len(algorithms))
    means = df["mean_cost"].values
    stds = df["std_cost"].values
    p95s = df["p95_cost"].values
    maxs = df["max_cost"].values

    bar_width = 0.5
    colors = [_color(a) for a in algorithms]

    # Bar for mean with error bars (std)
    bars = ax.bar(x, means, bar_width, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5,
                  label="Mean ± Std", alpha=0.85)

    # Markers for p95 and max
    ax.scatter(x, p95s, marker="^", color="black", s=80, zorder=5, label="95th Percentile")
    ax.scatter(x, maxs, marker="v", color="red", s=80, zorder=5, label="Maximum")

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Travel Cost", fontsize=12)
    ax.set_title("Monte Carlo Cost Distribution (1000 Samples, n=15)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig_monte_carlo.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Optimality Gap (heuristic vs exact MILP)
# ═══════════════════════════════════════════════════════════════════════════

def plot_optimality_gap():
    csv_path = os.path.join(RESULTS_DIR, "exp1_small.csv")
    if not os.path.exists(csv_path):
        print("  ✗ exp1_small.csv not found — skipping optimality gap")
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["worst_case_cost"])

    sizes = sorted(df["n"].unique())

    # Heuristics to compare against Robust MILP
    heuristics = ["NN", "NN+2opt", "BS-Heuristic", "Robust-2opt", "Robust-NN-2opt"]
    reference = "Robust MILP"

    rows = []
    for n in sizes:
        ref_df = df[(df["n"] == n) & (df["algorithm"] == reference)]
        if len(ref_df) == 0 or np.isnan(ref_df["worst_case_cost"].values[0]):
            continue
        ref_wc = ref_df["worst_case_cost"].values[0]
        for heur in heuristics:
            heur_df = df[(df["n"] == n) & (df["algorithm"] == heur)]
            if len(heur_df) == 0 or np.isnan(heur_df["worst_case_cost"].values[0]):
                continue
            heur_wc = heur_df["worst_case_cost"].values[0]
            gap = (heur_wc - ref_wc) / ref_wc if ref_wc > 0 else 0
            rows.append({"n": n, "algorithm": heur, "gap": gap})

    if not rows:
        print("  ✗ No data for optimality gap — skipping")
        return

    gap_df = pd.DataFrame(rows)
    n_heur = len(heuristics)
    valid_sizes = sorted(gap_df["n"].unique())
    n_sizes = len(valid_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / n_heur
    x = np.arange(n_sizes)

    for i, heur in enumerate(heuristics):
        h_df = gap_df[gap_df["algorithm"] == heur]
        vals = []
        for s in valid_sizes:
            subset = h_df[h_df["n"] == s]
            vals.append(subset["gap"].values[0] * 100 if len(subset) > 0 else 0)
        offset = (i - n_heur / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=heur, color=_color(heur),
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Instance Size (n)", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("Heuristic Optimality Gap vs Robust MILP", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in valid_sizes])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    _save(fig, "fig_optimality_gap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PS-10 Robust VRP — Generating Plots")
    print("=" * 60)
    print(f"  Reading from: {RESULTS_DIR}\n")

    plot_small_comparison()
    plot_scalability()
    plot_gamma_sensitivity()
    plot_xi_sensitivity()
    plot_monte_carlo()
    plot_optimality_gap()

    print("\n  All plots generated.")


if __name__ == "__main__":
    main()
