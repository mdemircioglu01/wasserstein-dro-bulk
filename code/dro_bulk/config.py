"""
Configuration objects for the multi-product bulk-transportation DRO model.

The naming follows the paper:
    I  : sources (origins)        -> n_sources
    J  : destinations             -> n_destinations
    K  : products                 -> n_products
    D  = J * K                    : dimension of the demand vector xi
    N                             : number of historical demand samples

A flat index d = j * K + k is used throughout to map a (destination, product)
pair (j, k) onto a single coordinate of the demand vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# --------------------------------------------------------------------------- #
#  Instance-size presets (Table: multi-scale instance design in the paper)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalePreset:
    name: str
    n_sources: int        # |I|
    n_destinations: int   # |J|
    n_products: int       # |K|


SCALE_PRESETS: dict[str, ScalePreset] = {
    "S": ScalePreset("S", n_sources=3,  n_destinations=5,  n_products=2),   # D = 10
    "M": ScalePreset("M", n_sources=6,  n_destinations=12, n_products=4),   # D = 48
    "L": ScalePreset("L", n_sources=12, n_destinations=25, n_products=6),   # D = 150
}


# --------------------------------------------------------------------------- #
#  Parameters that control how an instance is generated
# --------------------------------------------------------------------------- #
@dataclass
class InstanceConfig:
    scale: Literal["S", "M", "L"] = "S"

    # --- geography ---
    grid_km: float = 1000.0            # square side length for random coordinates

    # --- transportation cost:  c = (theta0 + theta1 * d) * rho * (1 + u) ---
    theta0: float = 5.0                # fixed handling component per unit
    theta1: float = 0.02              # per-km rate per ton-km
    cost_noise: float = 0.10           # u ~ U[0, cost_noise]

    # --- emission factor ef (CO2 per ton-km), negatively correlated with cost ---
    ef_low: float = 0.03               # clean / expensive modes
    ef_high: float = 0.18              # dirty / cheap modes
    ef_cost_correlation: float = -0.6  # target corr(cost-rate, ef) over links

    # --- product unit mass rho_k (tons per unit) ---
    rho_min: float = 0.5
    rho_max: float = 3.0

    # --- recourse multipliers (relative to regular cost / emission) ---
    mu_shortage_cost: float = 3.0      # mu_b  > 1 : emergency supply is expensive
    mu_surplus_cost: float = 0.5       # mu_h  < 1 : holding / disposal
    mu_shortage_emis: float = 2.5      # emergency supply is also dirty
    mu_surplus_emis: float = 0.8       # reverse logistics / disposal emission

    # --- supply / capacity ---
    supply_tightness: float = 1.20     # tau : sum supply = tau * expected demand
    link_capacity_factor: float = 0.5  # each link can carry at most this share
                                       # of total expected tonnage

    # --- fixed-charge link activation (NP-hard extension) ---
    # fixed cost / emission of opening a link, as a fraction of the cost/emission
    # of filling that link to capacity once.
    fixed_cost_factor: float = 0.25
    fixed_emis_factor: float = 0.25

    # --- demand distribution ---
    demand_mean_min: float = 20.0
    demand_mean_max: float = 100.0
    coeff_variation: float = 0.30      # cv in {0.1, 0.3, 0.5}
    demand_dist: Literal["normal", "lognormal"] = "lognormal"
    product_correlation: float = 0.4   # corr among products at the same destination
    support_sigma: float = 4.0         # upper bound = mean + support_sigma * std

    # --- sample sizes ---
    n_train: int = 50                  # N : historical samples seen by the model
    n_test: int = 2000                 # out-of-sample evaluation set

    seed: int = 0


# --------------------------------------------------------------------------- #
#  Parameters of the distributionally robust model
# --------------------------------------------------------------------------- #
@dataclass
class DROConfig:
    # Ground norm of the type-1 Wasserstein distance.
    #   "l1" -> dual norm l_inf -> linear program        (paper: type-1, LP)
    #   "l2" -> dual norm l_2   -> second-order cone prog (paper: type-2, SOCP)
    norm: Literal["l1", "l2"] = "l1"

    # Wasserstein radius epsilon (0 recovers the sample-average problem).
    epsilon: float = 0.0

    # Support handling:
    #   "box"  -> bounded demand support [lb, ub]; robustification changes the
    #             decision and yields the genuine LP / SOCP reformulation.
    #   "full" -> Xi = R^D; worst case reduces to SAA + epsilon * Lipschitz
    #             modulus (a constant shift), used only as a fast sanity baseline.
    support: Literal["box", "full"] = "box"

    # Row-generation control for the exact reformulation.
    max_outer_iter: int = 8
    tol: float = 1e-6

    # Separation oracle for the l2 box-support case (genuine row generation):
    #   "auto"       -> MISOCP if a MI-conic solver is installed, else fixedpoint
    #   "misocp"     -> exact, certified (needs GUROBI / MOSEK / SCIP / ECOS_BB)
    #   "fixedpoint" -> solver-agnostic multi-start heuristic separation
    #   "active"     -> legacy: subgradient at the sample point (fast, inexact)
    l2_separation: Literal["auto", "misocp", "fixedpoint", "active"] = "auto"
    fixedpoint_starts: int = 5

    # For the l1 box-support case use the exact compact separable LP (single
    # solve, no row generation). Set False to force the general row-generation
    # path (useful only for cross-validation; both give the same optimum).
    l1_compact: bool = True

    # Fixed-charge link activation: adds binary z_{ij} and fixed cost/emission,
    # turning the exact reformulation into a MILP (l1) / MISOCP (l2).
    fixed_charge: bool = False

    # cvxpy solver name (None lets cvxpy choose; e.g. "GUROBI", "MOSEK", "ECOS").
    solver: str | None = None
    verbose: bool = False
