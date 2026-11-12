"""
feasibility_checker.py — Validate route feasibility for TSP and CVRP.

Provides:
    - check_tsp_route: Verify that a route is a valid TSP tour.
    - check_cvrp_routes: Verify that a set of routes forms a valid CVRP solution.
"""

from typing import List, Tuple


# ---------------------------------------------------------------------------
# TSP route checker
# ---------------------------------------------------------------------------

def check_tsp_route(route: List[int], n: int) -> Tuple[bool, str]:
    """Check whether *route* is a valid TSP tour visiting customers 1..n.

    A valid TSP route:
        1. Starts and ends at the depot (node 0).
        2. Visits every customer 1, 2, …, n exactly once.
        3. Contains no repeated nodes (apart from the depot bookends).

    Parameters
    ----------
    route : list[int]
        Candidate route, e.g. ``[0, 3, 1, 2, 0]``.
    n : int
        Number of customers (customers are numbered 1..n).

    Returns
    -------
    (bool, str)
        ``(True, 'OK')`` if valid, otherwise ``(False, reason_string)``.
    """
    # --- Check 1: non-empty and depot bookends ---
    if len(route) < 2:
        return False, "Route has fewer than 2 nodes."
    if route[0] != 0:
        return False, f"Route does not start at depot 0 (starts at {route[0]})."
    if route[-1] != 0:
        return False, f"Route does not end at depot 0 (ends at {route[-1]})."

    # --- Check 2: correct length ---
    # Should be n+2 entries: depot + n customers + depot
    interior = route[1:-1]
    if len(interior) != n:
        return False, (
            f"Expected {n} customers in the interior, got {len(interior)}."
        )

    # --- Check 3: each customer 1..n appears exactly once, no repeats ---
    visited = set()
    for node in interior:
        if node < 1 or node > n:
            return False, f"Invalid customer index {node} (expected 1..{n})."
        if node in visited:
            return False, f"Customer {node} is visited more than once."
        visited.add(node)

    missing = set(range(1, n + 1)) - visited
    if missing:
        return False, f"Missing customers: {sorted(missing)}."

    return True, "OK"


# ---------------------------------------------------------------------------
# CVRP route checker
# ---------------------------------------------------------------------------

def check_cvrp_routes(
    routes: List[List[int]],
    n: int,
    demands: List[float],
    capacity: float,
) -> Tuple[bool, str]:
    """Check whether *routes* form a valid CVRP solution.

    Validity conditions:
        1. Each customer 1..n is visited exactly once across all routes.
        2. Every route starts and ends at the depot (node 0).
        3. The total demand served by each route does not exceed *capacity*.

    Parameters
    ----------
    routes : list[list[int]]
        List of vehicle routes.  Each route is ``[0, ..., 0]``.
    n : int
        Number of customers (1..n).
    demands : list[float]
        Demand of each node.  ``demands[0]`` (depot) is ignored.
    capacity : float
        Maximum vehicle capacity.

    Returns
    -------
    (bool, str)
        ``(True, 'OK')`` if valid, otherwise ``(False, reason_string)``.
    """
    if not routes:
        return False, "No routes provided."

    all_visited: List[int] = []

    for idx, route in enumerate(routes):
        # --- Depot bookends ---
        if len(route) < 2:
            return False, f"Route {idx} has fewer than 2 nodes."
        if route[0] != 0:
            return False, f"Route {idx} does not start at depot 0."
        if route[-1] != 0:
            return False, f"Route {idx} does not end at depot 0."

        # --- Capacity check ---
        interior = route[1:-1]
        route_demand = sum(demands[c] for c in interior)
        if route_demand > capacity + 1e-9:
            return False, (
                f"Route {idx} exceeds capacity: demand {route_demand:.2f} "
                f"> capacity {capacity:.2f}."
            )

        all_visited.extend(interior)

    # --- Coverage check ---
    visited_set = set(all_visited)
    for node in all_visited:
        if node < 1 or node > n:
            return False, f"Invalid customer index {node} in routes."

    # Duplicates
    if len(all_visited) != len(visited_set):
        from collections import Counter
        counts = Counter(all_visited)
        dupes = [c for c, cnt in counts.items() if cnt > 1]
        return False, f"Customers visited more than once: {sorted(dupes)}."

    # Missing customers
    missing = set(range(1, n + 1)) - visited_set
    if missing:
        return False, f"Missing customers: {sorted(missing)}."

    return True, "OK"


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== feasibility_checker.py — self-test ===\n")

    # ---- TSP checks ----
    n = 4

    # Valid route
    valid_route = [0, 1, 3, 2, 4, 0]
    ok, msg = check_tsp_route(valid_route, n)
    print(f"TSP  {valid_route}  →  valid={ok}  msg='{msg}'")
    assert ok, f"Expected valid: {msg}"

    # Missing customer
    bad_route1 = [0, 1, 3, 2, 0]
    ok1, msg1 = check_tsp_route(bad_route1, n)
    print(f"TSP  {bad_route1}  →  valid={ok1}  msg='{msg1}'")
    assert not ok1

    # Doesn't start at 0
    bad_route2 = [1, 2, 3, 4, 0]
    ok2, msg2 = check_tsp_route(bad_route2, n)
    print(f"TSP  {bad_route2}  →  valid={ok2}  msg='{msg2}'")
    assert not ok2

    # Duplicate customer
    bad_route3 = [0, 1, 1, 3, 4, 0]
    ok3, msg3 = check_tsp_route(bad_route3, n)
    print(f"TSP  {bad_route3}  →  valid={ok3}  msg='{msg3}'")
    assert not ok3

    # ---- CVRP checks ----
    demands = [0, 3, 5, 2, 4]  # depot demand = 0
    capacity = 8

    routes_valid = [[0, 1, 3, 0], [0, 2, 4, 0]]  # demands: 3+2=5, 5+4=9? NO
    ok4, msg4 = check_cvrp_routes(routes_valid, 4, demands, capacity)
    print(f"\nCVRP {routes_valid}  cap={capacity}  →  valid={ok4}  msg='{msg4}'")

    routes_ok = [[0, 1, 3, 0], [0, 2, 0], [0, 4, 0]]  # 5, 5, 4
    ok5, msg5 = check_cvrp_routes(routes_ok, 4, demands, capacity)
    print(f"CVRP {routes_ok}  cap={capacity}  →  valid={ok5}  msg='{msg5}'")
    assert ok5, f"Expected valid: {msg5}"

    print("\n✓ All feasibility_checker tests passed.")
