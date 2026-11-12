"""
baselines.py — Baseline TSP heuristics for the Robust VRP project.

Provides:
    - nearest_neighbour: Greedy nearest-neighbour construction heuristic.
    - two_opt_improve: 2-opt local search improvement.
    - nearest_neighbour_2opt: NN construction + 2-opt refinement.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from src.utils import route_cost


# ---------------------------------------------------------------------------
# Nearest-Neighbour heuristic
# ---------------------------------------------------------------------------

def nearest_neighbour(
    instance: Dict[str, Any],
    cost_matrix: Optional[np.ndarray] = None,
) -> List[int]:
    """Construct a TSP tour using the nearest-neighbour heuristic.

    Starting from the depot (node 0), repeatedly visit the nearest
    unvisited customer until all customers are served, then return to
    the depot.

    Parameters
    ----------
    instance : dict
        Problem instance (needs ``'n'`` and ``'nominal_costs'``).
    cost_matrix : np.ndarray or None
        Cost matrix to use.  Defaults to ``instance['nominal_costs']`` when
        *None*, but can be overridden (e.g. for Bertsimas-Sim reduction).

    Returns
    -------
    list[int]
        Route as ``[0, c1, c2, …, cn, 0]``.
    """
    n = instance['n']
    costs = cost_matrix if cost_matrix is not None else instance['nominal_costs']

    visited = np.zeros(n + 1, dtype=bool)
    visited[0] = True
    route = [0]

    current = 0
    for _ in range(n):
        # Find the nearest unvisited customer
        best_next = -1
        best_dist = float('inf')
        for j in range(1, n + 1):
            if not visited[j] and costs[current][j] < best_dist:
                best_dist = costs[current][j]
                best_next = j
        route.append(best_next)
        visited[best_next] = True
        current = best_next

    route.append(0)  # return to depot
    return route


# ---------------------------------------------------------------------------
# 2-opt local search
# ---------------------------------------------------------------------------

def two_opt_improve(
    route: List[int],
    cost_matrix: np.ndarray,
) -> List[int]:
    """Improve a TSP route using 2-opt local search until no improving swap.

    A 2-opt move removes two edges and reconnects the tour by reversing
    the segment between them.  We use the standard O(1) per-move
    evaluation for symmetric costs:

        improvement = c[i, i+1] + c[j, j+1] − c[i, j] − c[i+1, j+1]

    Parameters
    ----------
    route : list[int]
        Current tour, e.g. ``[0, 3, 1, 2, 0]``.
    cost_matrix : np.ndarray
        Symmetric cost matrix.

    Returns
    -------
    list[int]
        Improved (or unchanged) route.
    """
    route = list(route)  # work on a copy
    n = len(route) - 1   # number of stops (incl. depot at both ends minus one)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Current edges: (route[i-1], route[i]) and (route[j], route[j+1])
                # Proposed edges: (route[i-1], route[j]) and (route[i], route[j+1])
                c = cost_matrix
                old_cost = c[route[i - 1]][route[i]] + c[route[j]][route[j + 1]]
                new_cost = c[route[i - 1]][route[j]] + c[route[i]][route[j + 1]]

                if new_cost < old_cost - 1e-12:
                    # Reverse the segment route[i..j]
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True

    return route


# ---------------------------------------------------------------------------
# Combined NN + 2-opt
# ---------------------------------------------------------------------------

def nearest_neighbour_2opt(
    instance: Dict[str, Any],
    cost_matrix: Optional[np.ndarray] = None,
) -> List[int]:
    """Nearest-neighbour construction followed by 2-opt improvement.

    Parameters
    ----------
    instance : dict
        Problem instance.
    cost_matrix : np.ndarray or None
        Optional cost matrix override.

    Returns
    -------
    list[int]
        Improved route.
    """
    costs = cost_matrix if cost_matrix is not None else instance['nominal_costs']
    route = nearest_neighbour(instance, cost_matrix=costs)
    route = two_opt_improve(route, costs)
    return route


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== baselines.py — self-test ===\n")

    from src.instance_generator import generate_random_euclidean
    from src.utils import route_cost as rc

    inst = generate_random_euclidean(10, seed=7)
    costs = inst['nominal_costs']

    # 1. Nearest Neighbour
    nn_route = nearest_neighbour(inst)
    nn_cost = rc(nn_route, costs)
    print(f"NN route:      {nn_route}")
    print(f"NN cost:       {nn_cost:.4f}")

    # 2. 2-opt improvement
    opt_route = two_opt_improve(nn_route, costs)
    opt_cost = rc(opt_route, costs)
    print(f"\n2-opt route:   {opt_route}")
    print(f"2-opt cost:    {opt_cost:.4f}")
    print(f"Improvement:   {nn_cost - opt_cost:.4f}  ({100*(nn_cost-opt_cost)/nn_cost:.1f}%)")
    assert opt_cost <= nn_cost + 1e-9, "2-opt made the route worse!"

    # 3. Combined
    combo_route = nearest_neighbour_2opt(inst)
    combo_cost = rc(combo_route, costs)
    print(f"\nNN+2opt route: {combo_route}")
    print(f"NN+2opt cost:  {combo_cost:.4f}")

    # 4. Verify route validity
    from src.feasibility_checker import check_tsp_route
    for label, r in [('NN', nn_route), ('2-opt', opt_route), ('combo', combo_route)]:
        ok, msg = check_tsp_route(r, inst['n'])
        print(f"  {label} feasible: {ok} ({msg})")
        assert ok, f"{label} route is infeasible: {msg}"

    # 5. With a custom cost matrix
    modified_costs = costs + 0.5 * inst['max_delays']
    nn_mod = nearest_neighbour(inst, cost_matrix=modified_costs)
    print(f"\nNN (modified costs): {nn_mod}")
    print(f"  cost (modified): {rc(nn_mod, modified_costs):.4f}")
    print(f"  cost (nominal):  {rc(nn_mod, costs):.4f}")

    print("\n✓ All baselines tests passed.")
