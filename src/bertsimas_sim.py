"""
Bertsimas-Sim reduction for robust TSP.

Reduces the robust TSP to a polynomial number of nominal TSP sub-problems.
Each sub-problem uses modified arc costs, and the best solution across all
sub-problems is provably optimal for the robust problem.

Reference:
    Bertsimas, D. & Sim, M. (2003). Robust discrete optimization and
    network flows. Mathematical Programming, 98(1), 49-71.
"""

import time
import numpy as np
from typing import Callable, Optional

from src.utils import route_cost
from src.worst_case_evaluator import worst_case_cost, nominal_cost
from src.nominal_milp import solve_nominal_tsp_with_costs


def bertsimas_sim_exact(instance: dict, time_limit_per_sub: int = 60) -> dict:
    """Solve robust TSP via Bertsimas-Sim reduction (exact sub-problems).

    The reduction:
      1. Collect all distinct Δ_ij values, sort descending:
         δ_1 > δ_2 > ... > δ_r > 0.  Append δ_{r+1} = 0.
      2. For each l = 1, ..., r+1:
         - Build modified costs: c^(l)_ij = ĉ_ij + max(Δ_ij − δ_l, 0)
         - Solve nominal TSP with c^(l) using MILP
         - value_l = Γ · δ_l + OPT(NOM_l)
      3. Return the solution l* that minimises value_l.

    Args:
        instance: Problem instance dict.
        time_limit_per_sub: MILP time limit per sub-problem in seconds.

    Returns:
        dict with keys:
            'route', 'nominal_cost', 'worst_case_cost', 'best_delta',
            'status', 'solve_time'.
    """
    start = time.time()
    n = instance['n']
    c_hat = instance['nominal_costs']
    delta = instance['max_delays']
    gamma = instance['gamma']

    # Step 1: collect distinct positive Δ values, sorted descending
    delta_values = _get_sorted_delta_values(delta, n)

    best_value = float('inf')
    best_route = []
    best_delta_val = 0.0
    best_status = 'Not solved'

    # Step 2: solve each sub-problem
    for delta_l in delta_values:
        # Build modified cost matrix
        modified_costs = _build_modified_costs(c_hat, delta, delta_l, n)

        # Solve nominal TSP with modified costs
        result = solve_nominal_tsp_with_costs(instance, modified_costs,
                                               time_limit_per_sub)

        if result['route']:
            value_l = gamma * delta_l + result['cost']

            if value_l < best_value:
                best_value = value_l
                best_route = result['route']
                best_delta_val = delta_l
                best_status = result['status']

    solve_time = time.time() - start

    # Compute final costs using the evaluator
    nom_cost = nominal_cost(best_route, instance) if best_route else float('inf')
    wc_cost = worst_case_cost(best_route, instance) if best_route else float('inf')

    return {
        'route': best_route,
        'nominal_cost': nom_cost,
        'worst_case_cost': wc_cost,
        'best_delta': best_delta_val,
        'status': best_status,
        'solve_time': solve_time,
    }


def bertsimas_sim_heuristic(instance: dict,
                             heuristic_fn: Optional[Callable] = None) -> dict:
    """Solve robust TSP via Bertsimas-Sim reduction with a heuristic.

    Same reduction as bertsimas_sim_exact but replaces the exact MILP solver
    with a fast heuristic (default: nearest_neighbour_2opt from baselines).
    Much faster and scales to large instances.

    Args:
        instance: Problem instance dict.
        heuristic_fn: Optional callable.  Accepted signatures:
            - heuristic_fn(instance) → dict with 'route' and 'cost', or just a route list
            - heuristic_fn(instance, cost_matrix) → same
            If None, uses nearest_neighbour_2opt from baselines.

    Returns:
        dict with keys:
            'route', 'nominal_cost', 'worst_case_cost', 'best_delta',
            'status', 'solve_time'.
    """
    start = time.time()

    if heuristic_fn is None:
        from src.baselines import nearest_neighbour_2opt
        heuristic_fn = nearest_neighbour_2opt

    n = instance['n']
    c_hat = instance['nominal_costs']
    delta = instance['max_delays']
    gamma = instance['gamma']

    # Step 1: collect distinct positive Δ values
    delta_values = _get_sorted_delta_values(delta, n)

    best_value = float('inf')
    best_route = []
    best_delta_val = 0.0

    # Step 2: solve each sub-problem heuristically
    for delta_l in delta_values:
        modified_costs = _build_modified_costs(c_hat, delta, delta_l, n)

        # Build a temporary instance with modified costs for the heuristic
        modified_instance = dict(instance)
        modified_instance['nominal_costs'] = modified_costs

        # Try calling the heuristic with (instance, cost_matrix) first,
        # then fall back to (instance,) only
        route, cost = _call_heuristic(heuristic_fn, modified_instance,
                                       modified_costs)

        if route:
            value_l = gamma * delta_l + cost
            if value_l < best_value:
                best_value = value_l
                best_route = route
                best_delta_val = delta_l

    solve_time = time.time() - start

    nom_cost = nominal_cost(best_route, instance) if best_route else float('inf')
    wc_cost = worst_case_cost(best_route, instance) if best_route else float('inf')

    return {
        'route': best_route,
        'nominal_cost': nom_cost,
        'worst_case_cost': wc_cost,
        'best_delta': best_delta_val,
        'status': 'Heuristic',
        'solve_time': solve_time,
    }


# ── Helper functions ───────────────────────────────────────────────────

def _get_sorted_delta_values(delta: np.ndarray, n: int, max_thresholds: int = 25) -> list:
    """Collect distinct positive Δ_ij values sorted descending, plus 0.

    If the number of distinct values is large (e.g. > max_thresholds),
    selects an evenly-spaced subset of thresholds to ensure practical runtimes.

    Args:
        delta: Max delay matrix of shape (n+1, n+1).
        n: Number of customers.
        max_thresholds: Maximum number of distinct non-zero thresholds to test.

    Returns:
        List of floats: [δ_1, δ_2, ..., δ_r, 0.0] with δ_1 > δ_2 > ... > 0.
    """
    all_deltas = set()
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j and delta[i][j] > 0:
                all_deltas.add(round(float(delta[i][j]), 4))

    sorted_deltas = sorted(all_deltas, reverse=True)
    if len(sorted_deltas) > max_thresholds:
        indices = np.linspace(0, len(sorted_deltas) - 1, max_thresholds, dtype=int)
        sorted_deltas = [sorted_deltas[idx] for idx in indices]

    if not sorted_deltas or sorted_deltas[-1] != 0.0:
        sorted_deltas.append(0.0)
    return sorted_deltas


def _build_modified_costs(c_hat: np.ndarray, delta: np.ndarray,
                           delta_l: float, n: int) -> np.ndarray:
    """Build modified cost matrix: c^(l)_ij = ĉ_ij + max(Δ_ij − δ_l, 0).

    Args:
        c_hat: Nominal cost matrix.
        delta: Max delay matrix.
        delta_l: Current threshold value δ_l.
        n: Number of customers.

    Returns:
        Modified cost matrix as ndarray.
    """
    modified = c_hat.copy()
    excess = np.maximum(delta - delta_l, 0.0)
    modified = modified + excess
    return modified


def _call_heuristic(heuristic_fn: Callable, instance: dict,
                     cost_matrix: np.ndarray) -> tuple:
    """Call a heuristic function, handling different return types.

    Tries (instance, cost_matrix) first, then (instance,) only.

    Args:
        heuristic_fn: The heuristic callable.
        instance: Problem instance (with modified costs).
        cost_matrix: The modified cost matrix.

    Returns:
        Tuple of (route, cost).
    """
    import inspect

    result = None

    # Try two-argument call first
    try:
        sig = inspect.signature(heuristic_fn)
        params = list(sig.parameters.keys())
        if len(params) >= 2:
            result = heuristic_fn(instance, cost_matrix)
        else:
            result = heuristic_fn(instance)
    except TypeError:
        try:
            result = heuristic_fn(instance)
        except Exception:
            return ([], float('inf'))

    if result is None:
        return ([], float('inf'))

    # Handle dict return
    if isinstance(result, dict):
        route = result.get('route', [])
        cost = result.get('cost', route_cost(route, cost_matrix) if route else float('inf'))
        return (route, cost)

    # Handle list return (just a route)
    if isinstance(result, list):
        cost = route_cost(result, cost_matrix) if result else float('inf')
        return (result, cost)

    return ([], float('inf'))


# ── Quick self-test ────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.instance_generator import generate_random_euclidean

    print("=== Bertsimas-Sim Reduction Test ===\n")

    inst = generate_random_euclidean(n=6, seed=0)

    print("--- Exact (MILP sub-problems) ---")
    result_exact = bertsimas_sim_exact(inst, time_limit_per_sub=30)
    print(f"Status     : {result_exact['status']}")
    print(f"Route      : {result_exact['route']}")
    print(f"Nominal    : {result_exact['nominal_cost']:.2f}")
    print(f"Worst-case : {result_exact['worst_case_cost']:.2f}")
    print(f"Best δ     : {result_exact['best_delta']:.4f}")
    print(f"Time       : {result_exact['solve_time']:.2f}s")

    print("\n--- Heuristic (NN + 2-opt sub-problems) ---")
    result_heur = bertsimas_sim_heuristic(inst)
    print(f"Status     : {result_heur['status']}")
    print(f"Route      : {result_heur['route']}")
    print(f"Nominal    : {result_heur['nominal_cost']:.2f}")
    print(f"Worst-case : {result_heur['worst_case_cost']:.2f}")
    print(f"Best δ     : {result_heur['best_delta']:.4f}")
    print(f"Time       : {result_heur['solve_time']:.2f}s")
