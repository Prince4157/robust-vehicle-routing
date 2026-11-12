"""
Robust TSP MILP solver using the Bertsimas-Sim dual formulation.

Solves the Robust TSP where arc travel times are subject to budgeted
uncertainty.  The dual reformulation linearises the inner maximisation,
yielding a single compact MILP.
"""

import time
import numpy as np
import pulp

from src.utils import route_cost, route_edges
from src.worst_case_evaluator import worst_case_cost, nominal_cost


def solve_robust_tsp(instance: dict, time_limit: int = 300) -> dict:
    """Solve the robust TSP exactly via the Bertsimas-Sim dual MILP.

    Variables:
        x_ij ∈ {0,1}  – arc usage
        u_i           – MTZ position (customers only)
        θ  ≥ 0        – dual variable for the uncertainty budget
        p_ij ≥ 0      – per-arc surplus above θ

    Model:
        min  Σ ĉ_ij·x_ij + Γ·θ + Σ p_ij
        s.t.
          flow constraints (enter/leave once)
          MTZ subtour elimination
          p_ij ≥ Δ_ij·x_ij − θ   ∀ i≠j
          x_ij ∈ {0,1},  p_ij ≥ 0,  θ ≥ 0

    Args:
        instance: Problem instance dict with keys 'n', 'nominal_costs',
                  'max_delays', 'gamma'.
        time_limit: CBC solver time limit in seconds.

    Returns:
        dict with keys:
            'route', 'nominal_cost', 'worst_case_cost', 'theta',
            'status', 'solve_time'.
    """
    start = time.time()
    n = instance['n']
    N = range(n + 1)
    C = range(1, n + 1)
    c_hat = instance['nominal_costs']
    delta = instance['max_delays']
    gamma = instance['gamma']

    # ── Create LP ──────────────────────────────────────────────────────
    prob = pulp.LpProblem("RobustTSP", pulp.LpMinimize)

    # Binary arc variables
    x = {}
    for i in N:
        for j in N:
            if i != j:
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

    # MTZ position variables
    u = {}
    for i in C:
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n,
                                cat=pulp.LpContinuous)

    # Dual variable θ
    theta = pulp.LpVariable("theta", lowBound=0, cat=pulp.LpContinuous)

    # Surplus variables p_ij
    p = {}
    for i in N:
        for j in N:
            if i != j:
                p[i, j] = pulp.LpVariable(f"p_{i}_{j}", lowBound=0,
                                           cat=pulp.LpContinuous)

    # ── Objective ──────────────────────────────────────────────────────
    prob += (
        pulp.lpSum(c_hat[i][j] * x[i, j] for i in N for j in N if i != j)
        + gamma * theta
        + pulp.lpSum(p[i, j] for i in N for j in N if i != j),
        "RobustCost"
    )

    # ── Flow constraints ───────────────────────────────────────────────
    for i in N:
        prob += (
            pulp.lpSum(x[i, j] for j in N if j != i) == 1,
            f"leave_{i}"
        )
    for j in N:
        prob += (
            pulp.lpSum(x[i, j] for i in N if i != j) == 1,
            f"enter_{j}"
        )

    # ── MTZ subtour elimination ────────────────────────────────────────
    for i in C:
        for j in C:
            if i != j:
                prob += (
                    u[i] - u[j] + n * x[i, j] <= n - 1,
                    f"mtz_{i}_{j}"
                )

    # ── Robust counterpart constraints ─────────────────────────────────
    for i in N:
        for j in N:
            if i != j:
                prob += (
                    p[i, j] >= delta[i][j] * x[i, j] - theta,
                    f"robust_{i}_{j}"
                )

    # ── Solve ──────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)
    solve_time = time.time() - start

    status = pulp.LpStatus[prob.status]
    theta_val = pulp.value(theta) if pulp.value(theta) is not None else 0.0

    # ── Extract route ──────────────────────────────────────────────────
    route = _extract_route(x, n)

    # Compute costs using the evaluator for consistency
    nom_cost = nominal_cost(route, instance) if route else float('inf')
    wc_cost = worst_case_cost(route, instance) if route else float('inf')

    return {
        'route': route,
        'nominal_cost': nom_cost,
        'worst_case_cost': wc_cost,
        'theta': theta_val,
        'status': status,
        'solve_time': solve_time,
    }


def _extract_route(x: dict, n: int) -> list:
    """Follow arcs from depot to reconstruct the tour.

    Args:
        x: Dict of PuLP binary variables keyed by (i, j).
        n: Number of customers.

    Returns:
        List of node indices [0, ..., 0] forming the tour, or empty list
        if extraction fails.
    """
    N = range(n + 1)
    succ = {}
    for i in N:
        for j in N:
            if i != j and pulp.value(x[i, j]) is not None:
                if pulp.value(x[i, j]) > 0.5:
                    succ[i] = j

    if 0 not in succ:
        return []

    route = [0]
    current = succ.get(0)
    visited = {0}
    while current is not None and current not in visited:
        route.append(current)
        visited.add(current)
        current = succ.get(current)

    route.append(0)

    if len(route) != n + 2:
        return []

    return route


# ── Quick self-test ────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.instance_generator import generate_random_euclidean

    print("=== Robust MILP TSP Solver Test ===\n")

    inst = generate_random_euclidean(n=6, seed=0)
    result = solve_robust_tsp(inst, time_limit=60)

    print(f"Status     : {result['status']}")
    print(f"Route      : {result['route']}")
    print(f"Nominal    : {result['nominal_cost']:.2f}")
    print(f"Worst-case : {result['worst_case_cost']:.2f}")
    print(f"Theta      : {result['theta']:.4f}")
    print(f"Time       : {result['solve_time']:.2f}s")
