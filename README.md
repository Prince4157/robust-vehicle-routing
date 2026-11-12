# PS-10: Robust Vehicle Routing under Uncertain Travel Times

A comprehensive implementation and computational study of **robust optimization** for the Travelling Salesman Problem (TSP) with uncertain travel times, based on the **Bertsimas–Sim** budgeted uncertainty model.

## Problem Statement

In real-world vehicle routing, travel times are inherently uncertain due to traffic congestion, weather, and road conditions. The classical (nominal) TSP finds the minimum-cost route under fixed distances, but such solutions can perform poorly when actual travel times deviate from expected values.

This project formulates and solves the **Robust TSP** under the Bertsimas–Sim uncertainty set:

$$
\min_{\text{route}} \max_{\delta \in \mathcal{U}} \sum_{(i,j) \in \text{route}} (\hat{c}_{ij} + \delta_{ij})
$$

where the uncertainty set $\mathcal{U}$ allows at most $\Gamma$ edges to experience their worst-case delay $\Delta_{ij}$, and $\xi$ controls the relative magnitude of delays.

## Project Structure

```
ps10_robust_vrp/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/                               # Core algorithm library
│   ├── __init__.py
│   ├── utils.py                       # Common data structures and helpers
│   ├── instance_generator.py          # Problem instance creation (random & TSPLIB)
│   ├── feasibility_checker.py         # Route feasibility validation
│   ├── worst_case_evaluator.py        # Worst-case cost computation (primal & dual)
│   ├── baselines.py                   # Baseline heuristics (NN, 2-opt)
│   ├── nominal_milp.py               # Exact nominal TSP via MILP (MTZ formulation)
│   ├── robust_milp.py                # Exact robust TSP via MILP
│   ├── bertsimas_sim.py              # Bertsimas–Sim decomposition algorithm
│   ├── robust_2opt.py                # Proposed robust local search algorithms
│   └── monte_carlo.py                # Monte Carlo simulation evaluation
│
├── experiments/                       # Experiment scripts
│   ├── __init__.py
│   ├── run_all.py                     # Main experiment runner (generates CSVs)
│   └── plot_results.py                # Publication-quality plot generator
│
├── results/                           # Output directory (CSVs and PNGs)
│   ├── exp1_small.csv                 # Small instance comparison
│   ├── exp2_scalability.csv           # Scalability benchmarks
│   ├── exp3_gamma_sensitivity.csv     # Γ sensitivity analysis
│   ├── exp4_xi_sensitivity.csv        # ξ sensitivity analysis
│   ├── exp5_monte_carlo.csv           # Monte Carlo evaluation
│   ├── exp6_tsplib.csv                # TSPLIB benchmark results
│   ├── fig_small_comparison.png       # Bar chart of algorithm comparison
│   ├── fig_scalability.png            # Runtime scalability plot
│   ├── fig_gamma_sensitivity.png      # Γ sensitivity plot
│   ├── fig_xi_sensitivity.png         # ξ sensitivity plot
│   ├── fig_monte_carlo.png            # Monte Carlo distribution plot
│   └── fig_optimality_gap.png         # Optimality gap analysis
│
└── paper/                             # LaTeX report (if applicable)
```

## Installation

### Prerequisites

- Python 3.9 or later
- pip (Python package manager)

### Setup

```bash
# Clone or navigate to the project directory
cd ps10_robust_vrp

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package      | Version | Purpose                              |
|------------- |---------|--------------------------------------|
| numpy        | ≥ 1.24  | Numerical computation                |
| pandas       | ≥ 2.0   | Data manipulation and CSV I/O        |
| matplotlib   | ≥ 3.7   | Plotting and visualization           |
| networkx     | ≥ 3.1   | Graph algorithms                     |
| PuLP         | ≥ 2.7   | MILP modelling and solving           |
| scipy        | ≥ 1.10  | Scientific computing utilities       |

## Running Experiments

### Run all experiments

```bash
cd ps10_robust_vrp
python experiments/run_all.py
```

This will execute all six experiments sequentially and save CSV result files to the `results/` directory. Progress is printed to stdout.

**Estimated runtime:** 5–30 minutes depending on hardware (exact MILP solvers may take longer on larger instances).

### Individual experiments

Each experiment is implemented as a standalone function in `run_all.py`. You can import and run them individually:

```python
import sys
sys.path.insert(0, ".")
from experiments.run_all import experiment_1, experiment_3

df1 = experiment_1()   # Small instance comparison
df3 = experiment_3()   # Gamma sensitivity
```

## Generating Plots

After running the experiments (CSV files must exist in `results/`):

```bash
python experiments/plot_results.py
```

This generates six publication-quality PNG figures in the `results/` directory at 300 DPI.

## Experiments Overview

| # | Experiment | Parameters | Algorithms | Output |
|---|-----------|-----------|-----------|--------|
| 1 | Small Instance Comparison | n ∈ {5,8,10,12,15}, ξ=0.3 | All 8 | `exp1_small.csv` |
| 2 | Scalability | n ∈ {10,20,30,50,75,100}, ξ=0.3 | Heuristics only | `exp2_scalability.csv` |
| 3 | Γ Sensitivity | n=15, ξ=0.3, Γ ∈ {0,1,2,5,10,16} | Heuristics + Robust MILP | `exp3_gamma_sensitivity.csv` |
| 4 | ξ Sensitivity | n=15, Γ=5, ξ ∈ {0.0,0.1,0.2,0.3,0.5} | Heuristics + Robust MILP | `exp4_xi_sensitivity.csv` |
| 5 | Monte Carlo | n=15, ξ=0.3, Γ=5, 1000 samples | Selected methods | `exp5_monte_carlo.csv` |
| 6 | TSPLIB Benchmarks | burma14, ulysses16 | All applicable | `exp6_tsplib.csv` |

## Algorithms Implemented

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Nearest Neighbour (NN)** | Heuristic | Greedy construction heuristic |
| **NN + 2-opt** | Heuristic | NN with local search improvement (nominal objective) |
| **Nominal MILP** | Exact | MTZ formulation solved with PuLP/CBC |
| **Robust MILP** | Exact | Full robust counterpart MILP |
| **Bertsimas–Sim Exact** | Decomposition | B&S with MILP subproblems |
| **Bertsimas–Sim Heuristic** | Decomposition | B&S with NN+2opt subproblems |
| **Robust 2-opt** | Proposed | 2-opt using worst-case cost as objective |
| **Robust NN + 2-opt** | Proposed | Robust NN construction + robust 2-opt improvement |

## Module Descriptions

### `src/utils.py`
Core data structures (`create_instance`) and helper functions (`route_cost`, `route_edges`, `compute_distance_matrix`).

### `src/instance_generator.py`
Instance creation: random Euclidean instances, random delay instances, hardcoded TSPLIB benchmarks, and full instance suites.

### `src/feasibility_checker.py`
Route validation ensuring correctness (depot start/end, all customers visited, no repeats).

### `src/worst_case_evaluator.py`
Closed-form and dual formulations for computing the Bertsimas–Sim worst-case cost of a given route.

### `src/baselines.py`
Classical TSP heuristics: nearest-neighbour construction and 2-opt local search.

### `src/nominal_milp.py`
Exact optimal TSP via Miller–Tucker–Zemlin MILP formulation.

### `src/robust_milp.py`
Exact robust TSP: full MILP incorporating the Bertsimas–Sim uncertainty model.

### `src/bertsimas_sim.py`
Decomposition-based algorithm that iterates between master (TSP) and adversarial subproblems.

### `src/robust_2opt.py`
Proposed robust local search: 2-opt and nearest-neighbour heuristics that optimize the worst-case objective directly.

### `src/monte_carlo.py`
Simulation-based evaluation: sample random delay realizations and compute empirical cost distributions.

## References

1. **Bertsimas, D., & Sim, M.** (2003). Robust discrete optimization and network flows. *Mathematical Programming*, 98(1-3), 49–71.
2. **Bertsimas, D., & Sim, M.** (2004). The price of robustness. *Operations Research*, 52(1), 35–53.
3. **Miller, C. E., Tucker, A. W., & Zemlin, R. A.** (1960). Integer programming formulation of traveling salesman problems. *Journal of the ACM*, 7(4), 326–329.
4. **Montemanni, R., Barta, J., Mastrolilli, M., & Gambardella, L. M.** (2007). The robust traveling salesman problem with interval data. *Transportation Science*, 41(3), 366–381.
5. **Ordóñez, F., & Zhao, J.** (2007). Robust capacity expansion of network flows. *Networks*, 50(2), 136–145.

---

*PS-10 — Advanced Algorithms Course Project*
