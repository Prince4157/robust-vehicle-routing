#!/usr/bin/env python3
"""
run_all.py — Main experiment runner for PS-10 Robust Vehicle Routing.

Runs all experiments, collects results, and saves CSV files to the results/ directory.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow imports from the project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.utils import create_instance, route_cost, route_edges, compute_distance_matrix
from src.instance_generator import (
    generate_random_euclidean,
    generate_random_delays,
    load_tsplib_instance,
    generate_instance_suite,
)
from src.feasibility_checker import check_tsp_route
from src.worst_case_evaluator import worst_case_cost, worst_case_cost_dual, nominal_cost
from src.baselines import nearest_neighbour, two_opt_improve, nearest_neighbour_2opt
from src.nominal_milp import solve_nominal_tsp, solve_nominal_tsp_with_costs
from src.robust_milp import solve_robust_tsp
from src.bertsimas_sim import bertsimas_sim_exact, bertsimas_sim_heuristic
from src.robust_2opt import robust_2opt, robust_nearest_neighbour, robust_nn_2opt
from src.monte_carlo import evaluate_route_mc, compare_routes_mc

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run_algorithm(name, func, instance, timeout=300):
    """Run *func(instance)* with timing, returning (route, nom, wc, elapsed).

    On any failure the row is filled with NaN so the rest of the experiment
    can continue.
    """
    try:
        t0 = time.perf_counter()
        route = func(instance)
        elapsed = time.perf_counter() - t0
        if route is None:
            return None, np.nan, np.nan, elapsed
        nom = nominal_cost(route, instance)
        wc = worst_case_cost(route, instance)
        return route, nom, wc, elapsed
    except Exception as exc:
        print(f"    [WARN] {name} failed: {exc}")
        return None, np.nan, np.nan, np.nan


# --- Algorithm wrappers (each takes an instance, returns a route) ----------

def _alg_nn(inst):
    return nearest_neighbour(inst)


def _alg_nn_2opt(inst):
    return nearest_neighbour_2opt(inst)


def _extract_route(res):
    if res is None:
        return None
    if isinstance(res, dict):
        return res.get('route')
    if isinstance(res, (list, tuple)):
        return res[0] if isinstance(res[0], list) else list(res)
    return None


def _alg_nominal_milp(inst):
    res = solve_nominal_tsp(inst)
    return _extract_route(res)


def _alg_robust_milp(inst):
    res = solve_robust_tsp(inst)
    return _extract_route(res)


def _alg_bs_exact(inst):
    res = bertsimas_sim_exact(inst)
    return _extract_route(res)


def _alg_bs_heuristic(inst):
    res = bertsimas_sim_heuristic(inst)
    return _extract_route(res)


def _alg_robust_2opt(inst):
    res = robust_2opt(inst)
    return _extract_route(res)


def _alg_robust_nn_2opt(inst):
    res = robust_nn_2opt(inst)
    return _extract_route(res)


# Ordered list of ALL algorithms
ALL_ALGORITHMS = [
    ("NN",              _alg_nn),
    ("NN+2opt",         _alg_nn_2opt),
    ("Nominal MILP",    _alg_nominal_milp),
    ("Robust MILP",     _alg_robust_milp),
    ("BS-Exact",        _alg_bs_exact),
    ("BS-Heuristic",    _alg_bs_heuristic),
    ("Robust-2opt",     _alg_robust_2opt),
    ("Robust-NN-2opt",  _alg_robust_nn_2opt),
]

HEURISTIC_ALGORITHMS = [
    ("NN",              _alg_nn),
    ("NN+2opt",         _alg_nn_2opt),
    ("BS-Heuristic",    _alg_bs_heuristic),
    ("Robust-2opt",     _alg_robust_2opt),
    ("Robust-NN-2opt",  _alg_robust_nn_2opt),
]

HEURISTICS_PLUS_RMILP = [
    ("NN",              _alg_nn),
    ("NN+2opt",         _alg_nn_2opt),
    ("Robust MILP",     _alg_robust_milp),
    ("BS-Heuristic",    _alg_bs_heuristic),
    ("Robust-2opt",     _alg_robust_2opt),
    ("Robust-NN-2opt",  _alg_robust_nn_2opt),
]


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Small instances — compare ALL methods
# ═══════════════════════════════════════════════════════════════════════════

def experiment_1():
    """Small instances (n=5,8,10,12,15) with all algorithms."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Small Instance Comparison")
    print("=" * 70)

    sizes = [5, 8, 10, 12, 15]
    xi = 0.3
    seed = 42
    rows = []

    for n in sizes:
        gamma = n // 3
        inst = generate_random_euclidean(n, xi=xi, gamma=gamma, seed=seed)
        print(f"\n  n={n}, Γ={gamma}")

        for alg_name, alg_func in ALL_ALGORITHMS:
            route, nom, wc, elapsed = _run_algorithm(alg_name, alg_func, inst)
            rows.append({
                "n": n, "gamma": gamma, "xi": xi,
                "algorithm": alg_name,
                "nominal_cost": nom, "worst_case_cost": wc,
                "runtime": elapsed,
            })
            status = f"nom={nom:.2f}, wc={wc:.2f}, t={elapsed:.3f}s" if not np.isnan(nom) else "FAILED"
            print(f"    {alg_name:20s}  {status}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp1_small.csv")
    df.to_csv(path, index=False)
    print(f"\n  → Saved {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Scalability — heuristics only
# ═══════════════════════════════════════════════════════════════════════════

def experiment_2():
    """Scalability test (n=10..100) with heuristic algorithms only."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Scalability")
    print("=" * 70)

    sizes = [10, 20, 30, 50, 75, 100]
    xi = 0.3
    seed = 42
    rows = []

    for n in sizes:
        gamma = n // 3
        inst = generate_random_euclidean(n, xi=xi, gamma=gamma, seed=seed)
        print(f"\n  n={n}, Γ={gamma}")

        for alg_name, alg_func in HEURISTIC_ALGORITHMS:
            route, nom, wc, elapsed = _run_algorithm(alg_name, alg_func, inst)
            rows.append({
                "n": n, "gamma": gamma, "xi": xi,
                "algorithm": alg_name,
                "nominal_cost": nom, "worst_case_cost": wc,
                "runtime": elapsed,
            })
            status = f"nom={nom:.2f}, wc={wc:.2f}, t={elapsed:.3f}s" if not np.isnan(nom) else "FAILED"
            print(f"    {alg_name:20s}  {status}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp2_scalability.csv")
    df.to_csv(path, index=False)
    print(f"\n  → Saved {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Sensitivity to Γ (price of robustness)
# ═══════════════════════════════════════════════════════════════════════════

def experiment_3():
    """Fix n=15, ξ=0.3, vary Γ ∈ {0,1,2,5,10,n+1}."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sensitivity to Γ")
    print("=" * 70)

    n = 15
    xi = 0.3
    seed = 42
    gamma_values = [0, 1, 2, 5, 10, n + 1]
    rows = []

    for gamma in gamma_values:
        inst = generate_random_euclidean(n, xi=xi, gamma=gamma, seed=seed)
        print(f"\n  Γ={gamma}")

        for alg_name, alg_func in HEURISTICS_PLUS_RMILP:
            route, nom, wc, elapsed = _run_algorithm(alg_name, alg_func, inst)
            rows.append({
                "n": n, "gamma": gamma, "xi": xi,
                "algorithm": alg_name,
                "nominal_cost": nom, "worst_case_cost": wc,
                "runtime": elapsed,
            })
            status = f"nom={nom:.2f}, wc={wc:.2f}" if not np.isnan(nom) else "FAILED"
            print(f"    {alg_name:20s}  {status}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp3_gamma_sensitivity.csv")
    df.to_csv(path, index=False)
    print(f"\n  → Saved {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Sensitivity to ξ
# ═══════════════════════════════════════════════════════════════════════════

def experiment_4():
    """Fix n=15, Γ=5, vary ξ ∈ {0.0, 0.1, 0.2, 0.3, 0.5}."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Sensitivity to ξ (uncertainty level)")
    print("=" * 70)

    n = 15
    gamma = 5
    seed = 42
    xi_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows = []

    for xi in xi_values:
        inst = generate_random_euclidean(n, xi=xi, gamma=gamma, seed=seed)
        print(f"\n  ξ={xi}")

        for alg_name, alg_func in HEURISTICS_PLUS_RMILP:
            route, nom, wc, elapsed = _run_algorithm(alg_name, alg_func, inst)
            rows.append({
                "n": n, "gamma": gamma, "xi": xi,
                "algorithm": alg_name,
                "nominal_cost": nom, "worst_case_cost": wc,
                "runtime": elapsed,
            })
            status = f"nom={nom:.2f}, wc={wc:.2f}" if not np.isnan(nom) else "FAILED"
            print(f"    {alg_name:20s}  {status}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp4_xi_sensitivity.csv")
    df.to_csv(path, index=False)
    print(f"\n  → Saved {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Monte Carlo evaluation
# ═══════════════════════════════════════════════════════════════════════════

def experiment_5():
    """Monte Carlo simulation on n=15 instance with 1000 samples."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Monte Carlo Evaluation")
    print("=" * 70)

    n = 15
    xi = 0.3
    gamma = 5
    seed = 42
    n_samples = 1000

    inst = generate_random_euclidean(n, xi=xi, gamma=gamma, seed=seed)

    # Obtain routes from selected algorithms
    algorithms_for_mc = [
        ("NN+2opt",         _alg_nn_2opt),
        ("Robust MILP",     _alg_robust_milp),
        ("BS-Heuristic",    _alg_bs_heuristic),
        ("Robust-2opt",     _alg_robust_2opt),
    ]

    route_dict = {}
    for alg_name, alg_func in algorithms_for_mc:
        try:
            route = alg_func(inst)
            if route is not None:
                route_dict[alg_name] = route
                print(f"  Obtained route from {alg_name}")
            else:
                print(f"  [WARN] {alg_name} returned None")
        except Exception as exc:
            print(f"  [WARN] {alg_name} failed: {exc}")

    if not route_dict:
        print("  No routes obtained — skipping Monte Carlo evaluation.")
        return pd.DataFrame()

    # Run Monte Carlo comparison
    rows = []
    try:
        mc_results = compare_routes_mc(route_dict, inst, n_samples=n_samples, seed=seed)
        for alg_name, stats in mc_results.items():
            rows.append({
                "algorithm": alg_name,
                "mean_cost": stats.get("mean", np.nan),
                "std_cost": stats.get("std", np.nan),
                "p95_cost": stats.get("p95", np.nan),
                "max_cost": stats.get("max", np.nan),
                "min_cost": stats.get("min", np.nan),
                "nominal_cost": nominal_cost(route_dict[alg_name], inst),
                "worst_case_cost": worst_case_cost(route_dict[alg_name], inst),
            })
            print(f"  {alg_name:20s}  mean={stats.get('mean', np.nan):.2f}  "
                  f"std={stats.get('std', np.nan):.2f}  "
                  f"p95={stats.get('p95', np.nan):.2f}  "
                  f"max={stats.get('max', np.nan):.2f}")
    except Exception as exc:
        print(f"  [WARN] Monte Carlo comparison failed: {exc}")
        # Fallback: run individual evaluations
        for alg_name, route in route_dict.items():
            try:
                costs = evaluate_route_mc(route, inst, n_samples=n_samples, seed=seed)
                costs = np.array(costs)
                rows.append({
                    "algorithm": alg_name,
                    "mean_cost": np.mean(costs),
                    "std_cost": np.std(costs),
                    "p95_cost": np.percentile(costs, 95),
                    "max_cost": np.max(costs),
                    "min_cost": np.min(costs),
                    "nominal_cost": nominal_cost(route, inst),
                    "worst_case_cost": worst_case_cost(route, inst),
                })
                print(f"  {alg_name:20s}  mean={np.mean(costs):.2f}  "
                      f"std={np.std(costs):.2f}  "
                      f"p95={np.percentile(costs, 95):.2f}  "
                      f"max={np.max(costs):.2f}")
            except Exception as exc2:
                print(f"  [WARN] MC for {alg_name} failed: {exc2}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp5_monte_carlo.csv")
    df.to_csv(path, index=False)
    print(f"\n  → Saved {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: TSPLIB instances
# ═══════════════════════════════════════════════════════════════════════════

def experiment_6():
    """Run all applicable methods on TSPLIB benchmark instances."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: TSPLIB Instances")
    print("=" * 70)

    tsplib_names = ["burma14", "ulysses16"]
    rows = []

    for tsp_name in tsplib_names:
        try:
            inst = load_tsplib_instance(tsp_name)
        except Exception as exc:
            print(f"  [WARN] Could not load {tsp_name}: {exc}")
            continue

        n = inst["n"]
        print(f"\n  Instance: {tsp_name} (n={n})")

        for alg_name, alg_func in ALL_ALGORITHMS:
            route, nom, wc, elapsed = _run_algorithm(alg_name, alg_func, inst)
            rows.append({
                "instance": tsp_name, "n": n,
                "algorithm": alg_name,
                "nominal_cost": nom, "worst_case_cost": wc,
                "runtime": elapsed,
            })
            status = f"nom={nom:.2f}, wc={wc:.2f}, t={elapsed:.3f}s" if not np.isnan(nom) else "FAILED"
            print(f"    {alg_name:20s}  {status}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp6_tsplib.csv")
    df.to_csv(path, index=False)
    print(f"\n  → Saved {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PS-10: Robust Vehicle Routing — Full Experiment Suite          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Results will be saved to: {RESULTS_DIR}")

    t_start = time.perf_counter()

    df1 = experiment_1()
    df2 = experiment_2()
    df3 = experiment_3()
    df4 = experiment_4()
    df5 = experiment_5()
    df6 = experiment_6()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE  (total time: {total_time:.1f}s)")
    print("=" * 70)
    print(f"\nCSV files saved in: {RESULTS_DIR}")
    for fname in ["exp1_small.csv", "exp2_scalability.csv",
                   "exp3_gamma_sensitivity.csv", "exp4_xi_sensitivity.csv",
                   "exp5_monte_carlo.csv", "exp6_tsplib.csv"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname}  (missing)")


if __name__ == "__main__":
    main()
