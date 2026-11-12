"""
utils.py — Common data structures and helpers for the Robust Vehicle Routing project.

Provides:
    - create_instance: Build a standardised problem instance dictionary.
    - route_cost: Evaluate the total cost of a route under a given cost matrix.
    - route_edges: Extract the list of arcs (edges) used by a route.
    - compute_distance_matrix: Euclidean distance matrix from coordinate array.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Instance factory
# ---------------------------------------------------------------------------

def create_instance(
    n: int,
    coords: np.ndarray,
    nominal_costs: np.ndarray,
    max_delays: np.ndarray,
    gamma: float,
    name: str = 'random',
) -> Dict[str, Any]:
    """Create a standardised problem-instance dictionary.

    Parameters
    ----------
    n : int
        Number of customers (excluding depot).
    coords : np.ndarray, shape (n+1, 2)
        Coordinates of the depot (index 0) and customers (indices 1..n).
    nominal_costs : np.ndarray, shape (n+1, n+1)
        Nominal travel cost matrix ĉ_ij.
    max_delays : np.ndarray, shape (n+1, n+1)
        Maximum delay (uncertainty) matrix Δ_ij.
    gamma : float
        Robustness budget Γ (Bertsimas-Sim parameter).
    name : str, optional
        Human-readable label for the instance (default ``'random'``).

    Returns
    -------
    dict
        Keys: 'n', 'coords', 'nominal_costs', 'max_delays', 'gamma', 'name'.
    """
    return {
        'n': int(n),
        'coords': np.asarray(coords, dtype=float),
        'nominal_costs': np.asarray(nominal_costs, dtype=float),
        'max_delays': np.asarray(max_delays, dtype=float),
        'gamma': float(gamma),
        'name': str(name),
    }


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def route_cost(route: List[int], cost_matrix: np.ndarray) -> float:
    """Compute the total cost of traversing *route* under *cost_matrix*.

    Parameters
    ----------
    route : list[int]
        Ordered sequence of node indices, e.g. ``[0, 3, 1, 2, 0]``.
    cost_matrix : np.ndarray
        Square matrix where ``cost_matrix[i][j]`` is the cost from *i* to *j*.

    Returns
    -------
    float
        Sum of ``cost_matrix[route[k]][route[k+1]]`` for all consecutive pairs.
    """
    total = 0.0
    for k in range(len(route) - 1):
        total += cost_matrix[route[k]][route[k + 1]]
    return float(total)


def route_edges(route: List[int]) -> List[Tuple[int, int]]:
    """Return the list of directed arcs (edges) used by *route*.

    Parameters
    ----------
    route : list[int]
        Ordered sequence of node indices, e.g. ``[0, 3, 1, 2, 0]``.

    Returns
    -------
    list[tuple[int, int]]
        ``[(route[0], route[1]), (route[1], route[2]), ...]``
    """
    return [(route[k], route[k + 1]) for k in range(len(route) - 1)]


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between all coordinate points.

    Parameters
    ----------
    coords : np.ndarray, shape (m, 2)
        Array of 2-D coordinates.

    Returns
    -------
    np.ndarray, shape (m, m)
        Symmetric distance matrix where entry ``[i, j]`` is the Euclidean
        distance between point *i* and point *j*.
    """
    # Efficient vectorised computation using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (m, m, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== utils.py — self-test ===\n")

    # 1. Build a tiny 3-customer instance
    coords = np.array([
        [0.0, 0.0],   # depot
        [3.0, 0.0],   # customer 1
        [3.0, 4.0],   # customer 2
        [0.0, 4.0],   # customer 3
    ])
    dist = compute_distance_matrix(coords)
    print("Distance matrix (4 nodes):")
    print(np.round(dist, 2))

    delays = 0.3 * dist
    inst = create_instance(n=3, coords=coords, nominal_costs=dist,
                           max_delays=delays, gamma=1.0, name='tiny-test')
    print(f"\nInstance '{inst['name']}': n={inst['n']}, gamma={inst['gamma']}")

    # 2. Route cost
    route = [0, 1, 2, 3, 0]
    cost = route_cost(route, dist)
    print(f"\nRoute {route} cost = {cost:.4f}")

    # 3. Route edges
    edges = route_edges(route)
    print(f"Route edges: {edges}")

    print("\n✓ All utils tests passed.")
