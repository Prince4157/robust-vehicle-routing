"""
worst_case_evaluator.py — Robust worst-case cost evaluation (Bertsimas-Sim model).

Provides:
    - nominal_cost: Nominal travel cost along a route.
    - worst_case_cost: Closed-form worst-case cost (primal / combinatorial).
    - worst_case_cost_dual: Dual LP formulation (should match the primal).
"""

import math
import numpy as np
from typing import List, Dict, Any

from src.utils import route_cost, route_edges


# ---------------------------------------------------------------------------
# Nominal cost
# ---------------------------------------------------------------------------

def nominal_cost(route: List[int], instance: Dict[str, Any]) -> float:
    """Sum of nominal costs ĉ_ij along the route.

    Parameters
    ----------
    route : list[int]
        Ordered node sequence, e.g. ``[0, 3, 1, 2, 0]``.
    instance : dict
        Problem instance (must contain ``'nominal_costs'``).

    Returns
    -------
    float
        Total nominal cost.
    """
    return route_cost(route, instance['nominal_costs'])


# ---------------------------------------------------------------------------
# Worst-case cost — primal (closed-form)
# ---------------------------------------------------------------------------

def worst_case_cost(route: List[int], instance: Dict[str, Any]) -> float:
    """Compute the Bertsimas-Sim worst-case cost of *route* (closed-form).

    The adversary selects which arcs to delay.  At most Γ arcs may be
    simultaneously at their maximum delay.  The closed-form solution is:

        WC = Σ ĉ_ij  +  Σ_{k=1}^{⌊Γ⌋} Δ_{(k)}  +  (Γ − ⌊Γ⌋) · Δ_{(⌊Γ⌋+1)}

    where Δ_{(k)} denotes the k-th largest delay among arcs on the route.

    Parameters
    ----------
    route : list[int]
        Ordered node sequence.
    instance : dict
        Must contain ``'nominal_costs'``, ``'max_delays'``, ``'gamma'``.

    Returns
    -------
    float
        Worst-case total cost.
    """
    nom = nominal_cost(route, instance)
    gamma = instance['gamma']
    delays_matrix = instance['max_delays']

    # Collect the Δ_ij values for every arc on the route
    edges = route_edges(route)
    deltas = sorted(
        [delays_matrix[i][j] for i, j in edges],
        reverse=True,
    )

    if not deltas or gamma <= 0:
        return nom

    floor_gamma = int(math.floor(gamma))
    frac_gamma = gamma - floor_gamma

    # Sum the floor(Γ) largest delays
    protection = 0.0
    for k in range(min(floor_gamma, len(deltas))):
        protection += deltas[k]

    # Fractional part: add (Γ − ⌊Γ⌋) × next largest, if available
    if frac_gamma > 0 and floor_gamma < len(deltas):
        protection += frac_gamma * deltas[floor_gamma]

    return nom + protection


# ---------------------------------------------------------------------------
# Worst-case cost — dual LP formulation
# ---------------------------------------------------------------------------

def worst_case_cost_dual(route: List[int], instance: Dict[str, Any]) -> float:
    """Compute the worst-case cost via the Bertsimas-Sim *dual* formulation.

    The dual of the inner maximisation is:

        min_{θ ≥ 0}  Σ ĉ_ij  +  Γ·θ  +  Σ_{(i,j) ∈ route} max(Δ_ij − θ, 0)

    The optimal θ* is one of the Δ_ij values (or 0), so we evaluate at each
    candidate.  This should yield the same result as :func:`worst_case_cost`.

    Parameters
    ----------
    route : list[int]
        Ordered node sequence.
    instance : dict
        Must contain ``'nominal_costs'``, ``'max_delays'``, ``'gamma'``.

    Returns
    -------
    float
        Worst-case total cost (via dual).
    """
    nom = nominal_cost(route, instance)
    gamma = instance['gamma']
    delays_matrix = instance['max_delays']

    edges = route_edges(route)
    deltas = np.array([delays_matrix[i][j] for i, j in edges])

    if len(deltas) == 0 or gamma <= 0:
        return nom

    # Candidate θ values: each Δ_ij and 0
    candidates = np.concatenate([[0.0], deltas])
    candidates = np.unique(candidates)

    best = float('inf')
    for theta in candidates:
        obj = nom + gamma * theta + np.sum(np.maximum(deltas - theta, 0.0))
        if obj < best:
            best = obj

    return best


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== worst_case_evaluator.py — self-test ===\n")

    from src.instance_generator import generate_random_euclidean
    from src.baselines import nearest_neighbour

    inst = generate_random_euclidean(8, xi=0.3, seed=0)
    route = nearest_neighbour(inst)

    nc = nominal_cost(route, inst)
    wc_primal = worst_case_cost(route, inst)
    wc_dual = worst_case_cost_dual(route, inst)

    print(f"Instance: {inst['name']}")
    print(f"Route:    {route}")
    print(f"Nominal cost:          {nc:.4f}")
    print(f"Worst-case (primal):   {wc_primal:.4f}")
    print(f"Worst-case (dual):     {wc_dual:.4f}")
    print(f"Primal ≈ Dual?         {abs(wc_primal - wc_dual) < 1e-9}")
    assert abs(wc_primal - wc_dual) < 1e-9, "Primal and dual disagree!"

    # Edge case: Γ = 0 should equal nominal
    inst_zero = generate_random_euclidean(5, xi=0.3, gamma=0.0, seed=1)
    route_z = nearest_neighbour(inst_zero)
    nc_z = nominal_cost(route_z, inst_zero)
    wc_z = worst_case_cost(route_z, inst_zero)
    print(f"\nΓ=0 check: nominal={nc_z:.4f}, WC={wc_z:.4f}  equal={abs(nc_z - wc_z) < 1e-9}")
    assert abs(nc_z - wc_z) < 1e-9

    # Fractional Γ
    inst_frac = generate_random_euclidean(6, xi=0.4, gamma=2.5, seed=3)
    route_f = nearest_neighbour(inst_frac)
    wc_p = worst_case_cost(route_f, inst_frac)
    wc_d = worst_case_cost_dual(route_f, inst_frac)
    print(f"\nΓ=2.5 check: primal={wc_p:.4f}, dual={wc_d:.4f}  match={abs(wc_p - wc_d) < 1e-9}")
    assert abs(wc_p - wc_d) < 1e-9

    print("\n✓ All worst_case_evaluator tests passed.")
