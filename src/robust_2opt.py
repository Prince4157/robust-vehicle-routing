"""
robust_2opt.py — Proposed Robust 2-opt and Robust Nearest-Neighbour algorithms.

Directly optimizes the Bertsimas-Sim worst-case cost:
  WC(route) = nominal_cost + sum of ⌊Γ⌋ largest Δ on route + (Γ - ⌊Γ⌋) * next largest

Unlike nominal 2-opt which has O(1) move evaluation for symmetric costs,
robust 2-opt evaluates the worst-case cost across candidate moves.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional

from src.utils import route_cost, route_edges
from src.worst_case_evaluator import worst_case_cost, nominal_cost
from src.baselines import nearest_neighbour


class ResultDict(dict):
    """A dictionary that can also be unpacked as (route, worst_case_cost)."""
    def __iter__(self):
        return iter([self['route'], self['worst_case_cost']])


def robust_nearest_neighbour(instance: Dict[str, Any]) -> List[int]:
    """Greedy construction heuristic accounting for uncertainty.

    At each step, chooses the next customer j that minimizes the penalized
    effective cost:
        c_hat[current, j] + (gamma / (n + 1)) * delta[current, j]

    Parameters
    ----------
    instance : dict
        Problem instance containing 'n', 'nominal_costs', 'max_delays', 'gamma'.

    Returns
    -------
    list[int]
        Constructed tour [0, c_1, ..., c_n, 0].
    """
    n = instance['n']
    c_hat = instance['nominal_costs']
    delta = instance['max_delays']
    gamma = instance.get('gamma', n / 3.0)

    # Weight factor for delays
    weight = gamma / max(1.0, float(n + 1))

    visited = np.zeros(n + 1, dtype=bool)
    visited[0] = True
    route = [0]
    current = 0

    for _ in range(n):
        best_j = -1
        best_val = float('inf')
        for j in range(1, n + 1):
            if not visited[j]:
                effective_cost = c_hat[current][j] + weight * delta[current][j]
                if effective_cost < best_val:
                    best_val = effective_cost
                    best_j = j
        route.append(best_j)
        visited[best_j] = True
        current = best_j

    route.append(0)
    return route


def robust_2opt(
    instance: Dict[str, Any],
    initial_route: Optional[List[int]] = None,
    max_passes: int = 50,
) -> ResultDict:
    """Robust 2-opt local search directly minimizing worst-case cost.

    Parameters
    ----------
    instance : dict
        Problem instance.
    initial_route : list[int] or None
        Starting tour. If None, uses nearest_neighbour on nominal costs.
    max_passes : int
        Maximum number of outer improvement passes.

    Returns
    -------
    ResultDict
        Dictionary containing 'route', 'nominal_cost', 'worst_case_cost',
        'iterations', 'runtime'.
    """
    start_time = time.time()
    n = instance['n']

    if initial_route is None:
        current_route = list(nearest_neighbour(instance))
    else:
        current_route = list(initial_route)

    current_wc = worst_case_cost(current_route, instance)
    total_moves = 0
    num_nodes = len(current_route) - 1  # n + 1 nodes (depot appears twice)

    for pass_idx in range(max_passes):
        improved = False

        # Try all 2-opt segment reversals
        # route is [0, c_1, c_2, ..., c_n, 0]
        for i in range(1, num_nodes - 1):
            for j in range(i + 1, num_nodes):
                # Reverse segment between i and j inclusive
                cand_route = current_route[:i] + current_route[i:j + 1][::-1] + current_route[j + 1:]
                cand_wc = worst_case_cost(cand_route, instance)

                if cand_wc < current_wc - 1e-6:
                    current_route = cand_route
                    current_wc = cand_wc
                    total_moves += 1
                    improved = True
                    break  # First-improvement strategy for faster convergence
            if improved:
                break

        if not improved:
            break

    elapsed = time.time() - start_time
    nom = nominal_cost(current_route, instance)

    res = ResultDict({
        'route': current_route,
        'nominal_cost': nom,
        'worst_case_cost': current_wc,
        'iterations': total_moves,
        'runtime': elapsed,
    })
    return res


def robust_nn_2opt(instance: Dict[str, Any]) -> ResultDict:
    """Robust Nearest Neighbour followed by Robust 2-opt."""
    initial = robust_nearest_neighbour(instance)
    return robust_2opt(instance, initial_route=initial)


if __name__ == '__main__':
    from src.instance_generator import generate_random_euclidean
    inst = generate_random_euclidean(10, xi=0.3, gamma=3, seed=42)
    res = robust_2opt(inst)
    print("Robust 2-opt test:", res['worst_case_cost'], "route len:", len(res['route']))
    res2 = robust_nn_2opt(inst)
    print("Robust NN+2opt test:", res2['worst_case_cost'], "route len:", len(res2['route']))

