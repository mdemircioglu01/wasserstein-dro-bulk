"""
Multi-objective solution methods: the exact augmented epsilon-constraint front
and the NSGA-II approximation (pymoo).  Both optimise the *same* robust
objectives, so their Pareto fronts are directly comparable.
"""
from __future__ import annotations

import numpy as np

from .config import DROConfig
from .data_generation import Instance
from .dro_model import DROSolver
from .recourse import evaluate_plan


# --------------------------------------------------------------------------- #
#  Exact front via the augmented epsilon-constraint method
# --------------------------------------------------------------------------- #
def epsilon_constraint_front(inst: Instance, dro_cfg: DROConfig,
                             n_points: int = 11) -> list[dict]:
    """Trace the cost--emission Pareto front exactly (LP for l1, SOCP for l2)."""
    solver = DROSolver(inst, dro_cfg)

    cost_anchor = solver.solve_single("cost")        # min cost
    emis_anchor = solver.solve_single("emission")    # min emission
    e_lo = emis_anchor["emis_total"]                 # smallest achievable emission
    e_hi = cost_anchor["emis_total"]                 # emission at the cheapest plan

    front = [dict(cost_anchor, kappa=e_hi, anchor="min_cost"),
             dict(emis_anchor, kappa=e_lo, anchor="min_emission")]

    if e_hi > e_lo + 1e-9:
        for kappa in np.linspace(e_lo, e_hi, n_points)[1:-1]:
            sol = solver.solve_epsilon_constraint("cost", "emission", float(kappa))
            if sol.get("x") is not None:
                front.append(dict(sol, kappa=float(kappa), anchor=None))

    front = [f for f in front if f.get("x") is not None]
    front.sort(key=lambda f: f["cost_total"])
    return _nondominated(front)


def _nondominated(points: list[dict]) -> list[dict]:
    """Keep only Pareto-efficient (cost, emission) points."""
    keep = []
    for p in points:
        c, e = p["cost_total"], p["emis_total"]
        if not any((q["cost_total"] <= c + 1e-9) and (q["emis_total"] <= e + 1e-9)
                   and (q["cost_total"] < c - 1e-9 or q["emis_total"] < e - 1e-9)
                   for q in points):
            keep.append(p)
    keep.sort(key=lambda f: f["cost_total"])
    return keep


# --------------------------------------------------------------------------- #
#  Approximate front via NSGA-II
# --------------------------------------------------------------------------- #
def _repair(inst: Instance, x_ijk: np.ndarray, iters: int = 4) -> np.ndarray:
    """Project a raw plan onto the supply / link-capacity constraints."""
    x = np.maximum(x_ijk, 0.0)
    rho = inst.rho
    for _ in range(iters):
        # supply: sum_j x[i,j,k] <= S[i,k]
        used = x.sum(axis=1)                                  # (I, K)
        scale = np.minimum(1.0, inst.supply / np.maximum(used, 1e-12))
        x = x * scale[:, None, :]
        # link capacity: sum_k rho_k x[i,j,k] <= Q[i,j]
        ton = (x * rho[None, None, :]).sum(axis=2)            # (I, J)
        scale = np.minimum(1.0, inst.link_cap / np.maximum(ton, 1e-12))
        x = x * scale[:, :, None]
    return x


def _feasible_init(inst: Instance, rng) -> np.ndarray:
    """A feasible first-stage plan that ships toward a random fraction of mean
    demand, choosing sources by a random cost/emission preference, then repaired
    to supply/link feasibility.  The random preference spreads the initial
    population along the cost--emission trade-off (the trade-off here comes from
    choosing cheap/dirty vs. expensive/clean source-links)."""
    I, J, K = inst.n_sources, inst.n_destinations, inst.n_products
    x = np.zeros((I, J, K))
    remaining = inst.supply.copy()
    w = rng.uniform(0.0, 1.0)                          # 0 = green, 1 = cheap
    level = rng.uniform(0.2, 1.2, size=(J, K))         # target fraction of mean
    target = np.minimum(level * inst.demand_mean, inst.support_ub)
    score = w * inst.cost + (1.0 - w) * inst.emission  # (I,J,K), lower = better
    for k in range(K):
        for j in range(J):
            T = target[j, k]
            for i in np.argsort(score[:, j, k]):
                if T <= 1e-9:
                    break
                give = min(T, remaining[i, k])
                x[i, j, k] += give
                remaining[i, k] -= give
                T -= give
    return _repair(inst, x).reshape(-1)


def nsga2_front(inst: Instance, dro_cfg: DROConfig,
                pop_size: int = 60, n_gen: int = 60, seed: int = 0):
    """Approximate the Pareto front with NSGA-II, optimising the same robust
    objectives as the exact method.

    The population is seeded with feasible, demand-aware plans and the variables
    are bounded by ``min(supply, demand upper bound)``.  Both are essential for
    good coverage on the larger instances: a naive random start followed by a
    shrink-only repair systematically under-ships and is dominated everywhere.
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    I, J, K = inst.n_sources, inst.n_destinations, inst.n_products
    supply_b = np.repeat(inst.supply.reshape(I, 1, K), J, axis=1)        # (I,J,K)
    dem_b = np.repeat(inst.support_ub.reshape(1, J, K), I, axis=0)       # (I,J,K)
    xu = np.minimum(supply_b, dem_b).reshape(-1)

    rng = np.random.default_rng(seed)
    X0 = np.array([_feasible_init(inst, rng) for _ in range(pop_size)])
    X0 = np.minimum(X0, xu)

    class BulkProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=I * J * K, n_obj=2, n_constr=0,
                             xl=np.zeros(I * J * K), xu=xu)

        def _evaluate(self, xflat, out, *args, **kwargs):
            x = _repair(inst, xflat.reshape(I, J, K))
            ev = evaluate_plan(inst, x, dro_cfg)
            out["F"] = [ev["cost_total"], ev["emis_total"]]

    res = minimize(BulkProblem(), NSGA2(pop_size=pop_size, sampling=X0),
                   ("n_gen", n_gen), seed=seed, verbose=False)
    F = np.atleast_2d(res.F)
    order = np.argsort(F[:, 0])
    return F[order]
