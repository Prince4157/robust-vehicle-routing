"""
Exact nominal TSP solver using MTZ formulation with PuLP.

Solves the Travelling Salesman Problem exactly using Mixed-Integer Linear
Programming with Miller-Tucker-Zemlin subtour elimination constraints.
"""

import time
import numpy as np
import pulp

from src.utils import route_cost


def solve_nominal_tsp(instance: dict, time_limit: int = 300) -> dict:
    """Solve the nominal TSP exactly using MILP (MTZ subtour elimination).

    Uses the nominal cost matrix from the instance to find the minimum-cost
    Hamiltonian tour starting and ending at the depot (node 0).

    Args:
        instance: Problem instance dict with keys 'n', 'nominal_costs', etc.
        time_limit: Solver time limit in seconds (default 300).

    Returns:
        dict with keys:
            'route': list of node indices forming the tour (starts/ends at 0),
            'cost': float total nominal tour cost,
            'status': str solver status ('Optimal', 'Feasible', etc.),
            'solve_time': float wall-clock seconds spent solving.
    """
    cost_matrix = instance['nominal_costs']
    return solve_nominal_tsp_with_costs(instance, cost_matrix, time_limit)


def solve_nominal_tsp_with_costs(instance: dict, cost_matrix: np.ndarray,
                                  time_limit: int = 300) -> dict:
    """Solve the nominal TSP exactly using a custom cost matrix.

    This variant accepts an arbitrary cost matrix instead of using the
    instance's nominal costs.  It is required by the Bertsimas-Sim reduction
    which solves a sequence of nominal sub-problems with modified costs.

    Args:
        instance: Problem instance dict (needs 'n').
        cost_matrix: 2-D array of shape (n+1, n+1) with arc costs.
        time_limit: Solver time limit in seconds.

    Returns:
        dict with keys 'route', 'cost', 'status', 'solve_time'.
    """
    start = time.time()
    n = instance['n']  # number of customers (nodes 1..n)
    N = range(n + 1)   # all nodes including depot 0
    C = range(1, n + 1)  # customers only

    # ── Create the LP problem ──────────────────────────────────────────
    prob = pulp.LpProblem("NominalTSP", pulp.LpMinimize)

    # Decision variables
    x = {}
    for i in N:
        for j in N:
            if i != j:
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

    # MTZ position variables for customers (not depot)
    u = {}
    for i in C:
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n,
                                cat=pulp.LpContinuous)

    # ── Objective: minimise total cost ─────────────────────────────────
    prob += pulp.lpSum(
        cost_matrix[i][j] * x[i, j] for i in N for j in N if i != j
    ), "TotalCost"

    # ── Constraints ────────────────────────────────────────────────────
    # Each node is left exactly once
    for i in N:
        prob += (
            pulp.lpSum(x[i, j] for j in N if j != i) == 1,
            f"leave_{i}"
        )

    # Each node is entered exactly once
    for j in N:
        prob += (
            pulp.lpSum(x[i, j] for i in N if i != j) == 1,
            f"enter_{j}"
        )

    # MTZ subtour elimination (only for customer nodes)
    for i in C:
        for j in C:
            if i != j:
                prob += (
                    u[i] - u[j] + n * x[i, j] <= n - 1,
                    f"mtz_{i}_{j}"
                )

    # ── Solve ──────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)
    solve_time = time.time() - start

    status = pulp.LpStatus[prob.status]

    # ── Extract route ──────────────────────────────────────────────────
    route = _extract_route(x, n)
    cost = route_cost(route, cost_matrix) if route else float('inf')

    return {
        'route': route,
        'cost': cost,
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
        if no valid tour could be extracted.
    """
    N = range(n + 1)
    # Build successor map
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

    # Close the tour back to depot
    route.append(0)

    # Validate: must visit all n+1 nodes and return to depot
    if len(route) != n + 2:
        return []

    return route


# ── Quick self-test ────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.instance_generator import generate_random_euclidean

    print("=== Nominal MILP TSP Solver Test ===\n")

    inst = generate_random_euclidean(n=6, seed=0)
    result = solve_nominal_tsp(inst, time_limit=60)

    print(f"Status : {result['status']}")
    print(f"Route  : {result['route']}")
    print(f"Cost   : {result['cost']:.2f}")
    print(f"Time   : {result['solve_time']:.2f}s")

    # Test with custom cost matrix
    print("\n--- Custom cost matrix test ---")
    custom_costs = inst['nominal_costs'] * 1.1
    result2 = solve_nominal_tsp_with_costs(inst, custom_costs, time_limit=60)
    print(f"Status : {result2['status']}")
    print(f"Route  : {result2['route']}")
    print(f"Cost   : {result2['cost']:.2f}")
    print(f"Time   : {result2['solve_time']:.2f}s")
