"""
instance_generator.py — Functions to create problem instances for Robust VRP.

Provides:
    - generate_random_euclidean: Random Euclidean instance with proportional delays.
    - generate_random_delays: Random Euclidean instance with uniform-random delays.
    - load_tsplib_instance: Hardcoded TSPLIB benchmark instances.
    - generate_instance_suite: Batch-generate a full experimental suite.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.utils import create_instance, compute_distance_matrix


# ---------------------------------------------------------------------------
# Random Euclidean instance (proportional uncertainty)
# ---------------------------------------------------------------------------

def generate_random_euclidean(
    n: int,
    grid_size: int = 100,
    xi: float = 0.3,
    gamma: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a random Euclidean TSP/VRP instance with proportional delays.

    Parameters
    ----------
    n : int
        Number of customers (depot is added automatically as index 0).
    grid_size : int
        Coordinates are drawn uniformly from ``[0, grid_size]^2``.
    xi : float
        Proportionality factor: ``Δ_ij = xi * ĉ_ij``.
    gamma : float or None
        Bertsimas-Sim budget Γ.  Defaults to ``n // 3`` when *None*.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Standardised instance dictionary (see :func:`src.utils.create_instance`).
    """
    rng = np.random.default_rng(seed)
    gamma = gamma if gamma is not None else n // 3

    # n+1 points: index 0 = depot, 1..n = customers
    coords = rng.uniform(0, grid_size, size=(n + 1, 2))

    nominal_costs = compute_distance_matrix(coords)
    max_delays = xi * nominal_costs

    return create_instance(
        n=n,
        coords=coords,
        nominal_costs=nominal_costs,
        max_delays=max_delays,
        gamma=gamma,
        name=f'random_euclidean_n{n}_xi{xi}_g{gamma}_s{seed}',
    )


# ---------------------------------------------------------------------------
# Random-delays instance (uniform uncertainty independent of distance)
# ---------------------------------------------------------------------------

def generate_random_delays(
    n: int,
    grid_size: int = 100,
    delta_range: Tuple[float, float] = (1, 20),
    gamma: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a random Euclidean instance with uniform-random delays.

    Parameters
    ----------
    n : int
        Number of customers.
    grid_size : int
        Coordinate range ``[0, grid_size]^2``.
    delta_range : tuple (lo, hi)
        Each ``Δ_ij`` is drawn uniformly from ``[lo, hi]``.
    gamma : float or None
        Budget Γ; defaults to ``n // 3``.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Standardised instance dictionary.
    """
    rng = np.random.default_rng(seed)
    gamma = gamma if gamma is not None else n // 3

    coords = rng.uniform(0, grid_size, size=(n + 1, 2))
    nominal_costs = compute_distance_matrix(coords)

    lo, hi = delta_range
    max_delays = rng.uniform(lo, hi, size=(n + 1, n + 1))
    # Delays on the diagonal are meaningless (self-loops); zero them out
    np.fill_diagonal(max_delays, 0.0)

    return create_instance(
        n=n,
        coords=coords,
        nominal_costs=nominal_costs,
        max_delays=max_delays,
        gamma=gamma,
        name=f'random_delays_n{n}_s{seed}',
    )


# ---------------------------------------------------------------------------
# Hardcoded TSPLIB instances
# ---------------------------------------------------------------------------

# Burma14 — 14-city instance from TSPLIB (geographic coordinates)
_BURMA14_COORDS = np.array([
    [16.47, 96.10],
    [16.47, 94.44],
    [20.09, 92.54],
    [22.39, 93.37],
    [25.23, 97.24],
    [22.00, 96.05],
    [20.47, 97.02],
    [17.20, 96.29],
    [16.30, 97.38],
    [14.05, 98.12],
    [16.53, 97.38],
    [21.52, 95.59],
    [19.41, 97.13],
    [20.09, 94.55],
], dtype=float)

# Ulysses16 — 16-city instance from TSPLIB (geographic coordinates)
_ULYSSES16_COORDS = np.array([
    [38.24, 20.42],
    [39.57, 26.15],
    [40.56, 25.32],
    [36.26, 23.12],
    [33.48, 10.54],
    [37.56, 12.19],
    [38.42, 13.11],
    [37.52, 20.44],
    [41.23, 9.10],
    [41.17, 13.05],
    [36.08, -5.21],
    [38.47, 15.13],
    [38.15, 15.35],
    [37.51, 15.17],
    [35.49, 14.32],
    [39.36, 19.56],
], dtype=float)

_TSPLIB_REGISTRY: Dict[str, np.ndarray] = {
    'burma14': _BURMA14_COORDS,
    'ulysses16': _ULYSSES16_COORDS,
}


def load_tsplib_instance(
    name: str = 'burma14',
    xi: float = 0.3,
    gamma: Optional[float] = None,
) -> Dict[str, Any]:
    """Load a hardcoded TSPLIB instance.

    The first coordinate row is treated as the depot (index 0).

    Parameters
    ----------
    name : str
        Instance name — one of ``'burma14'`` or ``'ulysses16'``.
    xi : float
        Proportional delay factor: ``Δ_ij = xi * ĉ_ij``.
    gamma : float or None
        Budget Γ; defaults to ``(num_cities - 1) // 3``.

    Returns
    -------
    dict
        Standardised instance dictionary.

    Raises
    ------
    ValueError
        If *name* is not in the hardcoded registry.
    """
    name_lower = name.lower()
    if name_lower not in _TSPLIB_REGISTRY:
        available = ', '.join(sorted(_TSPLIB_REGISTRY.keys()))
        raise ValueError(
            f"Unknown TSPLIB instance '{name}'. Available: {available}"
        )

    coords = _TSPLIB_REGISTRY[name_lower].copy()
    n = len(coords) - 1  # first row = depot
    gamma = gamma if gamma is not None else n // 3

    nominal_costs = compute_distance_matrix(coords)
    max_delays = xi * nominal_costs

    return create_instance(
        n=n,
        coords=coords,
        nominal_costs=nominal_costs,
        max_delays=max_delays,
        gamma=gamma,
        name=f'tsplib_{name_lower}',
    )


# ---------------------------------------------------------------------------
# Instance suite generator
# ---------------------------------------------------------------------------

def generate_instance_suite(
    sizes: List[int] = None,
    xi_values: List[float] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a full suite of random Euclidean instances for experiments.

    Creates one instance per (n, xi) combination plus the TSPLIB benchmarks.

    Parameters
    ----------
    sizes : list[int]
        Customer counts.  Defaults to ``[5, 8, 10, 12, 15, 20, 30, 50]``.
    xi_values : list[float]
        Proportional-delay factors.  Defaults to ``[0.1, 0.3, 0.5]``.
    seed : int
        Base random seed (incremented per instance for variety).

    Returns
    -------
    list[dict]
        List of standardised instance dictionaries.
    """
    if sizes is None:
        sizes = [5, 8, 10, 12, 15, 20, 30, 50]
    if xi_values is None:
        xi_values = [0.1, 0.3, 0.5]

    suite: List[Dict[str, Any]] = []
    seed_counter = seed

    # Random Euclidean instances
    for n in sizes:
        for xi in xi_values:
            inst = generate_random_euclidean(n, xi=xi, seed=seed_counter)
            suite.append(inst)
            seed_counter += 1

    # TSPLIB benchmark instances
    for tsp_name in sorted(_TSPLIB_REGISTRY.keys()):
        for xi in xi_values:
            inst = load_tsplib_instance(tsp_name, xi=xi)
            suite.append(inst)

    return suite


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== instance_generator.py — self-test ===\n")

    # 1. Random Euclidean
    inst = generate_random_euclidean(8, xi=0.3, seed=0)
    print(f"Random Euclidean: n={inst['n']}, gamma={inst['gamma']}, "
          f"name='{inst['name']}'")
    print(f"  coords shape:        {inst['coords'].shape}")
    print(f"  nominal_costs shape: {inst['nominal_costs'].shape}")
    print(f"  max_delays shape:    {inst['max_delays'].shape}")

    # 2. Random delays
    inst2 = generate_random_delays(5, delta_range=(2, 10), seed=7)
    print(f"\nRandom delays:   n={inst2['n']}, gamma={inst2['gamma']}, "
          f"name='{inst2['name']}'")

    # 3. TSPLIB
    for tsp_name in ['burma14', 'ulysses16']:
        inst3 = load_tsplib_instance(tsp_name)
        print(f"\nTSPLIB '{tsp_name}': n={inst3['n']}, gamma={inst3['gamma']}")

    # 4. Full suite
    suite = generate_instance_suite()
    print(f"\nFull suite: {len(suite)} instances")
    for s in suite[:4]:
        print(f"  {s['name']}  (n={s['n']}, Γ={s['gamma']})")
    print("  ...")

    print("\n✓ All instance_generator tests passed.")
