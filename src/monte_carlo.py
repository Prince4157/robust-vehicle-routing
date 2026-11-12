"""
monte_carlo.py — Monte Carlo simulation and evaluation under travel time uncertainty.

Samples realisations of arc travel times under the Bertsimas-Sim uncertainty model:
  c_ij(ω) = ĉ_ij + z_ij * Δ_ij
where 0 <= z_ij <= 1 and at most Γ arcs experience delay simultaneously.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.utils import route_edges, route_cost
from src.worst_case_evaluator import worst_case_cost, nominal_cost


def sample_route_travel_times(
    route: List[int],
    instance: Dict[str, Any],
    num_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Sample actual travel times for a specific route across scenarios.

    In each scenario, a random subset of arcs on the route (up to Γ arcs)
    experiences a random delay fraction in [0, 1].

    Parameters
    ----------
    route : list[int]
        Route node sequence.
    instance : dict
        Problem instance.
    num_samples : int
        Number of Monte Carlo scenarios.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray of shape (num_samples,)
        Total travel time for the route under each scenario.
    """
    rng = np.random.default_rng(seed)
    edges = route_edges(route)
    m = len(edges)
    gamma = float(instance.get('gamma', 1.0))
    k = min(m, max(1, int(np.ceil(gamma))))

    c_hat = instance['nominal_costs']
    delta = instance['max_delays']

    # Nominal base cost
    base_nom = sum(c_hat[u][v] for u, v in edges)
    edge_deltas = np.array([delta[u][v] for u, v in edges])

    sample_costs = np.zeros(num_samples)

    for s in range(num_samples):
        # Nature randomly picks which arcs experience delay (up to k arcs)
        active_indices = rng.choice(m, size=min(k, m), replace=False)
        # Random fractional delays z_ij in [0, 1]
        z = rng.uniform(0.0, 1.0, size=len(active_indices))
        # Ensure budget constraint sum(z) <= gamma
        if np.sum(z) > gamma:
            z = z * (gamma / np.sum(z))

        total_delay = np.sum(edge_deltas[active_indices] * z)
        sample_costs[s] = base_nom + total_delay

    return sample_costs


def evaluate_route_mc(
    route: List[int],
    instance: Dict[str, Any],
    n_samples: int = 1000,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Evaluate travel times of a route under Monte Carlo simulations.

    Returns an array of travel time realizations.
    """
    samples = num_samples if num_samples is not None else n_samples
    return sample_route_travel_times(route, instance, num_samples=samples, seed=seed)


def compare_routes_mc(
    routes_dict: Dict[str, List[int]],
    instance: Dict[str, Any],
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple routes under identical Monte Carlo scenarios.

    Parameters
    ----------
    routes_dict : dict
        Mapping algorithm_name -> route list.
    instance : dict
        Problem instance.
    n_samples : int
        Number of simulation scenarios.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Mapping algorithm_name -> dict with statistics:
        'mean', 'std', 'p95', 'max', 'min', 'nominal', 'worst_case'.
    """
    results = {}
    for name, route in routes_dict.items():
        costs = evaluate_route_mc(route, instance, n_samples=n_samples, seed=seed)
        results[name] = {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs)),
            'p95': float(np.percentile(costs, 95)),
            'max': float(np.max(costs)),
            'min': float(np.min(costs)),
            'nominal': float(nominal_cost(route, instance)),
            'worst_case': float(worst_case_cost(route, instance)),
        }
    return results


if __name__ == '__main__':
    from src.instance_generator import generate_random_euclidean
    from src.baselines import nearest_neighbour
    inst = generate_random_euclidean(10, xi=0.3, gamma=3, seed=42)
    r = nearest_neighbour(inst)
    res = compare_routes_mc({'NN': r}, inst, n_samples=100)
    print("MC test results:", res)

